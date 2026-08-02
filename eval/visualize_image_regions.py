"""
5 панелей на UGC test-картинку: GT, SegFormer (до/после image-based
region fill), UNet (до/после) — тот же стиль, что visualize_wall_fill.py,
но граница ищется Canny-эджами на исходном фото, а не по предсказанной
wall-маске модели (eval/image_based_room_regions.py). Это дало ПОЛОЖИТЕЛЬНЫЙ
результат в отличие от wall_bounded_fill — особенно на room (UNet +0.02 F1).

Запуск:
    python eval/visualize_image_regions.py --all
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
from image_based_room_regions import detect_room_regions, majority_fill_by_regions  # noqa: E402
from wall_bounded_fill import ROOM_TYPE_IDS  # noqa: E402
from mask_nms import mask_nms  # noqa: E402
from visualize_wall_fill import (  # noqa: E402
    overlay_from_label_map, draw_ignore_hatch, legend_handles,
    draw_boundary_contour, diff_only_overlay,
)

CANNY_LO, CANNY_HI, DILATE_ITERS, MIN_AREA_FRAC = 80, 200, 2, 0.01


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
    ap.add_argument("--out-dir", default="docs/report_assets/image_region_comparison")
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

        regions = detect_room_regions(image_bgr, CANNY_LO, CANNY_HI, DILATE_ITERS, MIN_AREA_FRAC)
        boundary = regions == 0  # то, что НЕ вошло ни в один region — граница/шум

        panels = [("GT", gt_overlay_for(coco, img_id, h, w, room_by_img, ignore_by_img, image_bgr))]
        for model_key, label in (("segformer", "SegFormer"), ("unet_baseline", "UNet-simple")):
            pmap = pred_label_map(preds_by_model[model_key], img_id, h, w, args.score_thresh)

            before_panel = overlay_from_label_map(image_bgr, pmap)
            before_panel = draw_boundary_contour(before_panel, boundary, thickness=2)
            panels.append((f"{label} — до (контур = Canny)", before_panel))

            filled = majority_fill_by_regions(pmap, regions, args.min_room_frac)

            # панель 1: ГДЕ заполнилось — чистое фото + розовый акцент, без цветов классов
            where_panel = diff_only_overlay(image_bgr, pmap, filled, ROOM_TYPE_IDS)
            where_panel = draw_boundary_contour(where_panel, boundary, thickness=2)
            panels.append((f"{label} — где заполнилось", where_panel))

            # панель 2: КАК заполнилось — финальные цвета классов, без розового поверх
            after_panel = overlay_from_label_map(image_bgr, filled)
            after_panel = draw_boundary_contour(after_panel, boundary, thickness=2)
            panels.append((f"{label} — как заполнилось (классы)", after_panel))

        fig, axes = plt.subplots(1, 7, figsize=(28, 4.5))
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
        print(f"[visualize_image_regions] -> {out_path}")


if __name__ == "__main__":
    main()
