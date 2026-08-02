"""
Коллаж GT vs RF-DETR vs UNet(+Canny-fill) на нашем held-out val-сплите
(ResPlan/CubiCasa) — иллюстрирует находку: RF-DETR (чекпоинт эпохи 4/5 из
100 запланированных) на этом домене коллапсирует в предсказание почти
везде класса "opening" (см. docs/experiments_log.md), хотя локализация
(class-agnostic IoU) в целом рабочая.

Запуск:
    python eval/visualize_valid_predictions.py --source resplan --n 6
    python eval/visualize_valid_predictions.py --source cubicasa --n 6
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from mask_nms import mask_nms  # noqa: E402
from compute_valid_metrics import gt_label_map, pred_label_map  # noqa: E402
from image_based_room_regions import detect_room_regions, majority_fill_by_regions  # noqa: E402
from visualize_wall_fill import overlay_from_label_map, legend_handles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["resplan", "cubicasa"], required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="docs/report_assets/valid_split_comparison")
    args = ap.parse_args()

    paths = load_paths()
    gt_path = Path(paths["derived"]["data_dir"]) / "valid_coco.json"
    images_root = Path(paths["combined_out_root"]) / "images"
    output_dir = Path(paths["derived"]["output_dir"])

    coco = COCO(str(gt_path))
    img_ids = coco.getImgIds()
    img_info = {i: coco.loadImgs([i])[0] for i in img_ids}
    img_wh = {i: (info["height"], info["width"]) for i, info in img_info.items()}
    src_ids = [i for i in img_ids if img_info[i]["file_name"].startswith(args.source + "/")]

    rng = random.Random(args.seed)
    # берём картинки с достаточным числом GT-аннотаций (не пустые/тривиальные)
    src_ids_with_anns = [i for i in src_ids if len(coco.imgToAnns.get(i, [])) >= 4]
    sample = rng.sample(src_ids_with_anns, min(args.n, len(src_ids_with_anns)))

    with open(output_dir / "rfdetr_seg_valid" / "predictions" / "valid_predictions.json", encoding="utf-8") as f:
        rfdetr_preds = json.load(f)
    rfdetr_preds = mask_nms(rfdetr_preds, img_wh, iou_thresh=0.5)
    rfdetr_by_img = {}
    for p in rfdetr_preds:
        rfdetr_by_img.setdefault(p["image_id"], []).append(p)

    with open(output_dir / "unet_baseline_valid" / "predictions" / "valid_predictions.json", encoding="utf-8") as f:
        unet_preds = json.load(f)
    unet_by_img = {}
    for p in unet_preds:
        unet_by_img.setdefault(p["image_id"], []).append(p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in sample:
        info = img_info[img_id]
        h, w = info["height"], info["width"]
        image_bgr = cv2.imread(str(images_root / info["file_name"]))

        gtm = gt_label_map(coco, img_id, h, w)
        rfdetr_pm = pred_label_map(rfdetr_by_img, img_id, h, w)
        unet_pm = pred_label_map(unet_by_img, img_id, h, w)
        regions = detect_room_regions(image_bgr, 80, 200, 2, 0.01)
        unet_filled = majority_fill_by_regions(unet_pm, regions, 0.3)

        panels = [
            ("GT", overlay_from_label_map(image_bgr, gtm)),
            ("RF-DETR (thresh=0.02, NMS 0.5)", overlay_from_label_map(image_bgr, rfdetr_pm)),
            ("UNet (сырой)", overlay_from_label_map(image_bgr, unet_pm)),
            ("UNet + Canny-fill", overlay_from_label_map(image_bgr, unet_filled)),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        for ax, (title, panel) in zip(axes, panels):
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(f"{info['file_name']}  (val-сплит, source={args.source})", fontsize=9)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=8,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1))

        stem = Path(info["file_name"]).stem
        out_path = out_dir / f"{args.source}_{stem}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"[visualize_valid_predictions] -> {out_path}")


if __name__ == "__main__":
    main()
