"""
Одна картинка = GT (сырой UGC, все категории) + предсказания ВСЕХ переданных
моделей рядом, панель за панелью.

Запуск:
    python eval/visualize_multi.py --gt data/ugc_test/test_coco.json \
        --raw-ugc-root "C:/Users/user/Downloads/avito-toilet/ugc_labeled" \
        --model rfdetr=output/rfdetr_seg/predictions/test_predictions.json \
        --model yolo=output/yolo_seg/predictions/test_predictions.json \
        --image-id 1 --out output/viz/img01_all_models.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

PALETTE = {
    "living": (66, 133, 244), "bedroom": (219, 68, 55), "bathroom": (244, 180, 0),
    "kitchen": (15, 157, 88), "balcony": (171, 71, 188), "wall": (120, 120, 120),
    "opening": (0, 172, 193),
    "room": (0, 0, 0), "hall": (140, 90, 40), "coridor": (90, 60, 140),
    "stairs": (0, 90, 200), "storage": (60, 60, 180), "toilet": (200, 200, 0),
    "restroom": (244, 180, 0), "door": (0, 172, 193), "window": (0, 172, 193),
    "enterence": (0, 172, 193),
}


def poly_to_mask(seg, h, w) -> np.ndarray:
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def draw_masks(base: np.ndarray, items: list[tuple[np.ndarray, str, str]]) -> np.ndarray:
    out = base.copy()
    overlay = base.copy()
    for mask, _, cls in items:
        overlay[mask] = PALETTE.get(cls, (255, 255, 255))
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    for mask, label, cls in items:
        color = PALETTE.get(cls, (255, 255, 255))
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        cv2.putText(out, label, (max(0, cx - 20), max(15, cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, label, (max(0, cx - 20), max(15, cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return out


def add_title(panel: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(panel, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="merged data/ugc_test/test_coco.json (для маппинга file_name/image_id)")
    ap.add_argument("--raw-ugc-root", required=True)
    ap.add_argument("--model", action="append", required=True,
                     help="имя=путь_к_test_predictions.json, можно несколько раз")
    ap.add_argument("--image-id", type=int, required=True)
    ap.add_argument("--score-thr", type=float, default=0.1)
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = json.load(open(args.gt, "r", encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in gt["categories"]}
    img_rec = next(im for im in gt["images"] if im["id"] == args.image_id)
    images_root = Path(args.images_root) if args.images_root else Path(args.gt).parent / "images"
    image_bgr = cv2.imread(str(images_root / img_rec["file_name"]))
    h, w = img_rec["height"], img_rec["width"]

    # --- GT panel (сырые UGC-лейблы, все категории) ---
    split, orig_name = img_rec["file_name"].split("__", 1)
    raw = json.load(open(Path(args.raw_ugc_root) / split / "_annotations.coco.json", "r", encoding="utf-8"))
    raw_cat_names = {c["id"]: c["name"] for c in raw["categories"]}
    raw_img = next(im for im in raw["images"] if im["file_name"] == orig_name)
    gt_items = []
    for ann in raw["annotations"]:
        if ann["image_id"] != raw_img["id"]:
            continue
        cls = raw_cat_names[ann["category_id"]]
        mask = poly_to_mask(ann["segmentation"], raw_img["height"], raw_img["width"])
        gt_items.append((mask, cls, cls))
    panels = [add_title(draw_masks(image_bgr, gt_items), "GT (raw UGC)")]

    # --- по панели на модель ---
    for spec in args.model:
        name, pred_path = spec.split("=", 1)
        preds = json.load(open(pred_path, "r", encoding="utf-8"))
        items = []
        for p in preds:
            if p["image_id"] != args.image_id or p["score"] < args.score_thr:
                continue
            cls = cat_names[p["category_id"]]
            mask = mask_utils.decode(p["segmentation"]).astype(bool)
            items.append((mask, f"{cls} {p['score']:.2f}", cls))
        panels.append(add_title(draw_masks(image_bgr, items), f"{name} ({len(items)})"))

    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    combined = panels[0]
    for p in panels[1:]:
        combined = np.hstack([combined, sep, p])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    print(f"image_id={args.image_id} file_name={img_rec['file_name']} -> {out_path}")


if __name__ == "__main__":
    main()
