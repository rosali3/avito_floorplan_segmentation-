"""
5 панелей на каждую UGC test-картинку: GT, SegFormer (до/после squareness-
фильтра), UNet (до/после). В отличие от visualize_wall_fill.py — тут не
трогаем стены, просто выбрасываем предсказанные room-type блобы с
подозрительно вытянутой формой (см. eval/room_shape_filter.py).

Лучшие пороги по эксперименту (F1 конкретно на классе "room"):
  UNet-simple: max_aspect=4.0 (F1 room 0.568->0.569)
  SegFormer:   max_aspect=2.5 (F1 room 0.507->0.512)

Запуск:
    python eval/visualize_room_filter.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from compute_confusion_matrix import gt_label_map, pred_label_map, split_ignore_regions, true_ignore_mask  # noqa: E402
from room_shape_filter import squareness_filter  # noqa: E402
from mask_nms import mask_nms  # noqa: E402
from visualize_wall_fill import overlay_from_label_map, draw_ignore_hatch, legend_handles  # noqa: E402

BEST_ASPECT = {"segformer": 2.5, "unet_baseline": 4.0}


def gt_overlay_for(coco, img_id, h, w, room_by_img, ignore_by_img, image_bgr):
    gmap = gt_label_map(coco, img_id, h, w, room_by_img.get(img_id, []))
    overlay = overlay_from_label_map(image_bgr, gmap)
    ig = true_ignore_mask(ignore_by_img, img_id, h, w)
    return draw_ignore_hatch(overlay, ig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--image-substr", default=None)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--out-dir", default="docs/report_assets/room_filter_comparison")
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

    preds_by_model = {}
    for model_key in ("segformer", "unet_baseline"):
        with open(output_dir / model_key / "predictions" / "test_predictions.json", encoding="utf-8") as f:
            preds = json.load(f)
        preds = [p for p in preds if p.get("score", 1.0) >= args.score_thresh]
        preds = mask_nms(preds, img_wh, iou_thresh=args.nms_iou)
        preds_by_model[model_key] = preds

    all_imgs = coco.loadImgs(coco.getImgIds())
    if not args.all:
        all_imgs = [im for im in all_imgs if args.image_substr in im["file_name"]]
        if not all_imgs:
            raise SystemExit("картинка не найдена")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_info in all_imgs:
        img_id, h, w = img_info["id"], img_info["height"], img_info["width"]
        image_bgr = cv2.imread(str(ugc_test_dir / "images" / img_info["file_name"]))
        if image_bgr is None:
            continue

        panels = [("GT", gt_overlay_for(coco, img_id, h, w, room_by_img, ignore_by_img, image_bgr))]
        for model_key, label in (("segformer", "SegFormer"), ("unet_baseline", "UNet-simple")):
            pmap = pred_label_map(preds_by_model[model_key], img_id, h, w, args.score_thresh)
            panels.append((f"{label} — до", overlay_from_label_map(image_bgr, pmap)))
            filtered = squareness_filter(pmap, max_aspect=BEST_ASPECT[model_key])
            panels.append((f"{label} — после (aspect<={BEST_ASPECT[model_key]})",
                           overlay_from_label_map(image_bgr, filtered)))

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
        for ax, (title, panel) in zip(axes, panels):
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(img_info["file_name"], fontsize=9)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=8,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1))

        stem = Path(img_info["file_name"]).stem
        out_path = out_dir / f"{stem}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"[visualize_room_filter] -> {out_path}")


if __name__ == "__main__":
    main()
