"""
Коллаж "до/после mask-NMS" для RF-DETR: на низком score-threshold (0.1)
RF-DETR сильно недокалиброван и выдаёт много дублирующих, почти
полностью перекрывающихся масок на одном и том же объекте (см.
eval/mask_nms.py). Здесь наглядно показываем, сколько инстансов реально
"схлопывается" NMS'ом и как это меняет картинку.

3 панели: GT | RF-DETR до NMS (каждый инстанс — своя обводка, чтобы были
видны дубли) | RF-DETR после NMS (iou_thresh=0.5).

Запуск:
    python eval/visualize_nms_effect.py --model rfdetr_seg --all
    python eval/visualize_nms_effect.py --model rfdetr_seg --image-substr MyfC7La4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from mask_nms import mask_nms  # noqa: E402
from compute_confusion_matrix import gt_label_map, split_ignore_regions, true_ignore_mask  # noqa: E402
from visualize_wall_fill import overlay_from_label_map, draw_ignore_hatch, legend_handles  # noqa: E402

# яркая качественная палитра для обводок отдельных инстансов (не связана с классом)
OUTLINE_COLORS = [
    (255, 0, 0), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 165, 0),
    (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128), (255, 0, 128),
]

FILL_PALETTE = {
    1: (255, 99, 71), 2: (60, 179, 113), 3: (65, 105, 225), 4: (255, 215, 0),
    5: (238, 130, 238), 6: (128, 128, 128), 7: (255, 140, 0),
}


def _decode(pred: dict, h: int, w: int) -> np.ndarray:
    seg = pred["segmentation"]
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def draw_instances(image_bgr: np.ndarray, preds: list[dict], h: int, w: int,
                    alpha: float = 0.35, show_score: bool = True) -> np.ndarray:
    """Каждый инстанс — полупрозрачная заливка цветом класса + СВОЯ обводка
    (циклический яркий цвет по порядку) — перекрывающиеся дубли видно по
    нескольким разноцветным контурам в одном месте."""
    out = image_bgr.astype(np.float32).copy()
    for i, pred in enumerate(preds):
        m = _decode(pred, h, w)
        if not m.any():
            continue
        cls = pred["category_id"]
        color = np.array(FILL_PALETTE.get(cls, (200, 200, 200))[::-1], dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + color * alpha
    out = out.astype(np.uint8)
    for i, pred in enumerate(preds):
        m = (_decode(pred, h, w).astype(np.uint8)) * 255
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = OUTLINE_COLORS[i % len(OUTLINE_COLORS)][::-1]
        cv2.drawContours(out, contours, -1, outline, 2)
        if show_score and contours:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                score = pred.get("score", 1.0)
                cv2.putText(out, f"{score:.2f}", (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, f"{score:.2f}", (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def gt_overlay_for(coco, img_id, h, w, room_by_img, ignore_by_img, image_bgr):
    gmap = gt_label_map(coco, img_id, h, w, room_by_img.get(img_id, []))
    overlay = overlay_from_label_map(image_bgr, gmap)
    ig = true_ignore_mask(ignore_by_img, img_id, h, w)
    return draw_ignore_hatch(overlay, ig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="напр. rfdetr_seg или rfdetr_seg_fullaug")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--image-substr", default=None)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if not args.all and not args.image_substr:
        raise SystemExit("укажи --all или --image-substr")

    paths = load_paths()
    ugc_test_dir = Path(paths["derived"]["ugc_test_dir"])
    output_dir = Path(paths["derived"]["output_dir"])
    gt_path = ugc_test_dir / "test_coco.json"

    coco = COCO(str(gt_path))
    room_by_img, ignore_by_img = split_ignore_regions(gt_path)
    img_wh = {im["id"]: (im["height"], im["width"]) for im in coco.loadImgs(coco.getImgIds())}

    with open(output_dir / args.model / "predictions" / "test_predictions.json", encoding="utf-8") as f:
        preds = json.load(f)
    preds = [p for p in preds if p.get("score", 1.0) >= args.score_thresh]
    preds_after = mask_nms(preds, img_wh, iou_thresh=args.nms_iou)

    preds_before_by_img: dict[int, list[dict]] = {}
    preds_after_by_img: dict[int, list[dict]] = {}
    for p in preds:
        preds_before_by_img.setdefault(p["image_id"], []).append(p)
    for p in preds_after:
        preds_after_by_img.setdefault(p["image_id"], []).append(p)

    all_imgs = coco.loadImgs(coco.getImgIds())
    if not args.all:
        all_imgs = [im for im in all_imgs if args.image_substr in im["file_name"]]
        if not all_imgs:
            raise SystemExit("картинка не найдена")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"docs/report_assets/nms_effect_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_info in all_imgs:
        img_id, h, w = img_info["id"], img_info["height"], img_info["width"]
        image_bgr = cv2.imread(str(ugc_test_dir / "images" / img_info["file_name"]))
        if image_bgr is None:
            continue

        before = preds_before_by_img.get(img_id, [])
        after = preds_after_by_img.get(img_id, [])
        n_removed = len(before) - len(after)

        gt_panel = gt_overlay_for(coco, img_id, h, w, room_by_img, ignore_by_img, image_bgr)
        before_panel = draw_instances(image_bgr, before, h, w)
        after_panel = draw_instances(image_bgr, after, h, w)

        panels = [
            ("GT", gt_panel),
            (f"RF-DETR ДО NMS ({len(before)} инстансов)", before_panel),
            (f"RF-DETR ПОСЛЕ NMS ({len(after)} инстансов, -{n_removed})", after_panel),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, (title, panel) in zip(axes, panels):
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(f"{img_info['file_name']}  |  {args.model}, score>={args.score_thresh}, "
                      f"NMS iou_thresh={args.nms_iou}", fontsize=9)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=8,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1))

        stem = Path(img_info["file_name"]).stem
        out_path = out_dir / f"{stem}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"[visualize_nms_effect] {img_info['file_name']}: {len(before)} -> {len(after)} (-{n_removed}) -> {out_path}")


if __name__ == "__main__":
    main()
