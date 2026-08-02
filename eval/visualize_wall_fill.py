"""
5 панелей на каждую UGC test-картинку: GT, SegFormer (до/после
wall-bounded-fill), UNet (до/после). Наглядно показывает эксперимент,
который в eval/wall_bounded_fill.py дал отрицательный результат по метрикам —
здесь видно ПОЧЕМУ (утечки через дырявые стены сливают соседние комнаты).

Запуск:
    python eval/visualize_wall_fill.py --all
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
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from compute_confusion_matrix import gt_label_map, pred_label_map, split_ignore_regions, true_ignore_mask  # noqa: E402
from wall_bounded_fill import wall_bounded_fill, WALL_ID, OPENING_ID, ROOM_TYPE_IDS  # noqa: E402
from mask_nms import mask_nms  # noqa: E402

PALETTE = {
    1: (255, 99, 71), 2: (60, 179, 113), 3: (65, 105, 225), 4: (255, 215, 0),
    5: (238, 130, 238), 6: (128, 128, 128), 7: (255, 140, 0), 8: (139, 69, 19),
}
CLASS_NAMES = {1: "living", 2: "bedroom", 3: "bathroom", 4: "kitchen",
               5: "balcony", 6: "wall", 7: "opening", 8: "room (только GT)"}

BOUNDARY_LINE_COLOR = (0, 0, 0)     # чёрный контур — где алгоритм видит wall+opening
ACCENT_FILL_COLOR = (255, 0, 255)   # неоновый розовый (BGR) — пиксели, дозаполненные wall-fill


def draw_boundary_contour(image_bgr: np.ndarray, boundary_mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Жирный контур по границе (wall+opening из предсказания) — где алгоритм
    "видит" стену/проём, из которых строится замкнутая область."""
    out = image_bgr.copy()
    mask_u8 = (boundary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, BOUNDARY_LINE_COLOR, thickness)
    return out


def highlight_filled_pixels(base_bgr: np.ndarray, before_map: np.ndarray, after_map: np.ndarray,
                             room_type_ids: list[int], alpha: float = 0.85) -> np.ndarray:
    """Суперакцентная заливка ТОЛЬКО тех пикселей, которые wall-fill реально
    добавил/перекрасил в room-тип (было что-то другое, стало room-класс)."""
    out = base_bgr.astype(np.float32).copy()
    changed = (before_map != after_map) & np.isin(after_map, room_type_ids)
    if changed.any():
        color = np.array(ACCENT_FILL_COLOR, dtype=np.float32)
        out[changed] = out[changed] * (1 - alpha) + color * alpha
    return out.astype(np.uint8)


def diff_only_overlay(image_bgr: np.ndarray, before_map: np.ndarray, after_map: np.ndarray,
                       room_type_ids: list[int], alpha: float = 0.9) -> np.ndarray:
    """То же самое, но поверх ЧИСТОГО фото (без цветов классов) — отдельная
    панель "где заполнилось", без примеси того, "чем" заполнилось."""
    return highlight_filled_pixels(image_bgr, before_map, after_map, room_type_ids, alpha=alpha)


def legend_handles():
    return [mpatches.Patch(color=tuple(c / 255 for c in PALETTE[cid]), label=name)
            for cid, name in CLASS_NAMES.items()]


def overlay_from_label_map(image_bgr: np.ndarray, label_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = image_bgr.astype(np.float32).copy()
    for cid, color in PALETTE.items():
        m = label_map == cid
        if not m.any():
            continue
        bgr = np.array(color[::-1], dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + bgr * alpha
    return out.astype(np.uint8)


def draw_ignore_hatch(image_bgr: np.ndarray, ignore_mask: np.ndarray, spacing: int = 8) -> np.ndarray:
    if not ignore_mask.any():
        return image_bgr
    out = image_bgr.copy()
    h, w = ignore_mask.shape
    hatch = np.zeros((h, w), dtype=bool)
    for offset in range(-h, w, spacing):
        ys = np.arange(h)
        xs = ys + offset
        valid = (xs >= 0) & (xs < w)
        hatch[ys[valid], xs[valid]] = True
    m = ignore_mask & hatch
    out[m] = (out[m].astype(np.float32) * 0.3 + np.array([40, 40, 40], dtype=np.float32) * 0.7).astype(np.uint8)
    dim = ignore_mask & ~hatch
    out[dim] = (out[dim].astype(np.float32) * 0.75).astype(np.uint8)
    return out


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
    ap.add_argument("--min-room-frac", type=float, default=0.3)
    ap.add_argument("--out-dir", default="docs/report_assets/wall_fill_comparison")
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
            boundary = np.isin(pmap, [WALL_ID, OPENING_ID])

            before_panel = overlay_from_label_map(image_bgr, pmap)
            before_panel = draw_boundary_contour(before_panel, boundary, thickness=3)
            panels.append((f"{label} — до (контур = wall+opening)", before_panel))

            filled = wall_bounded_fill(pmap, min_room_frac=args.min_room_frac)
            after_panel = overlay_from_label_map(image_bgr, filled)
            after_panel = highlight_filled_pixels(after_panel, pmap, filled, ROOM_TYPE_IDS)
            after_panel = draw_boundary_contour(after_panel, boundary, thickness=3)
            panels.append((f"{label} — после (розовым = дозаполнено)", after_panel))

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
        for ax, (title, panel) in zip(axes, panels):
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(img_info["file_name"], fontsize=9)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=len(CLASS_NAMES),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1))

        stem = Path(img_info["file_name"]).stem
        out_path = out_dir / f"{stem}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"[visualize_wall_fill] -> {out_path}")


if __name__ == "__main__":
    main()
