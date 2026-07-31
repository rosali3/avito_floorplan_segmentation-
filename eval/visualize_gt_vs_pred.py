"""
GT (слева) vs предсказание (справа), обе поверх оригинальной картинки.
Каждый класс — свой цвет (одинаковый на обеих панелях), полупрозрачная заливка
маски + подпись класса (+ score для предсказаний).

Запуск:
    python eval/visualize_gt_vs_pred.py --gt data/ugc_test/test_coco.json \
        --pred output/rfdetr_seg/predictions/test_predictions.json \
        --image-id 5 --out output/viz/img5_rfdetr.png
    # без --image-id берёт первую картинку, где есть хоть один GT-инстанс класса --highlight-class
    python eval/visualize_gt_vs_pred.py --gt ... --pred ... --highlight-class kitchen --out ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

PALETTE = {
    "living":   (66, 133, 244),
    "bedroom":  (219, 68, 55),
    "bathroom": (244, 180, 0),
    "kitchen":  (15, 157, 88),
    "balcony":  (171, 71, 188),
    "wall":     (120, 120, 120),
    "opening":  (0, 172, 193),
    # исключённые из официального GT категории UGC — отдельные "приглушённые" цвета
    "room":     (0, 0, 0),
    "hall":     (140, 90, 40),
    "coridor":  (90, 60, 140),
    "stairs":   (0, 90, 200),
    "storage":  (60, 60, 180),
    "toilet":   (200, 200, 0),
    "restroom": (244, 180, 0),
    "door":     (0, 172, 193),
    "window":   (0, 172, 193),
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
    """items: [(mask_bool, label_text, class_name), ...]"""
    out = base.copy()
    overlay = base.copy()
    for mask, _, cls in items:
        color = PALETTE.get(cls, (255, 255, 255))
        overlay[mask] = color
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--image-id", type=int, default=None)
    ap.add_argument("--highlight-class", default=None,
                     help="если --image-id не задан, берём первую картинку с GT-инстансом этого класса")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--images-root", default=None, help="по умолчанию <папка gt>/images")
    ap.add_argument("--raw-ugc-root", default=None,
                     help="если задан — слева рисуются СЫРЫЕ GT-лейблы UGC (включая "
                          "room/hall/coridor/stairs/storage, исключённые из officialного теста), "
                          "а не отфильтрованные из --gt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = json.load(open(args.gt, "r", encoding="utf-8"))
    preds = json.load(open(args.pred, "r", encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in gt["categories"]}

    image_id = args.image_id
    if image_id is None:
        target_cat_id = None
        if args.highlight_class:
            target_cat_id = {v: k for k, v in cat_names.items()}[args.highlight_class]
        for ann in gt["annotations"]:
            if target_cat_id is None or ann["category_id"] == target_cat_id:
                image_id = ann["image_id"]
                break
        if image_id is None:
            raise SystemExit("не нашёл подходящую картинку")

    img_rec = next(im for im in gt["images"] if im["id"] == image_id)
    images_root = Path(args.images_root) if args.images_root else Path(args.gt).parent / "images"
    image_bgr = cv2.imread(str(images_root / img_rec["file_name"]))
    h, w = img_rec["height"], img_rec["width"]

    gt_items = []
    gt_title = "GROUND TRUTH"
    if args.raw_ugc_root:
        # file_name в merged test_coco.json = "<split>__<original_name>"
        split, orig_name = img_rec["file_name"].split("__", 1)
        raw_path = Path(args.raw_ugc_root) / split / "_annotations.coco.json"
        raw = json.load(open(raw_path, "r", encoding="utf-8"))
        raw_cat_names = {c["id"]: c["name"] for c in raw["categories"]}
        raw_img = next(im for im in raw["images"] if im["file_name"] == orig_name)
        for ann in raw["annotations"]:
            if ann["image_id"] != raw_img["id"]:
                continue
            cls = raw_cat_names[ann["category_id"]]
            mask = poly_to_mask(ann["segmentation"], raw_img["height"], raw_img["width"])
            gt_items.append((mask, cls, cls))
        gt_title = "GROUND TRUTH (raw UGC, все категории)"
    else:
        for ann in gt["annotations"]:
            if ann["image_id"] != image_id:
                continue
            cls = cat_names[ann["category_id"]]
            mask = poly_to_mask(ann["segmentation"], h, w)
            gt_items.append((mask, cls, cls))

    pred_items = []
    for p in preds:
        if p["image_id"] != image_id or p["score"] < args.score_thr:
            continue
        cls = cat_names[p["category_id"]]
        mask = mask_utils.decode(p["segmentation"]).astype(bool)
        label = f"{cls} {p['score']:.2f}"
        pred_items.append((mask, label, cls))

    left = draw_masks(image_bgr, gt_items)
    right = draw_masks(image_bgr, pred_items)
    cv2.putText(left, gt_title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(left, gt_title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "PREDICTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(right, "PREDICTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    combined = np.hstack([left, sep, right])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    print(f"image_id={image_id} file_name={img_rec['file_name']} "
          f"gt_instances={len(gt_items)} pred_instances={len(pred_items)}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
