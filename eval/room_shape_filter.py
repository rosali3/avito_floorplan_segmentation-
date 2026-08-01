"""
Другая идея, чем wall_bounded_fill.py: не пытаться реконструировать комнату
через (ненадёжную) маску стен, а просто фильтровать САМИ предсказанные блобы
классов-комнат (living/bedroom/bathroom/kitchen/balcony) по форме — если
компонента сильно вытянута (похожа на полосу/утечку, а не на комнату),
выбрасываем её как шум. "Квадратность" меряем через aspect ratio
минимального повёрнутого прямоугольника (cv2.minAreaRect): ближе к 1 —
похоже на комнату, намного больше 1 — вытянутая полоса.

Запуск:
    python eval/room_shape_filter.py --model unet_baseline --max-aspect 2.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from compute_confusion_matrix import (  # noqa: E402
    gt_label_map, pred_label_map, split_ignore_regions, true_ignore_mask, ROOM_ID,
)
from wall_bounded_fill import evaluate_dense, summarize  # noqa: E402
from mask_nms import mask_nms  # noqa: E402

ROOM_TYPE_IDS = [1, 2, 3, 4, 5]


def component_aspect_ratio(comp_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("inf")
    cnt = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)
    if w == 0 or h == 0:
        return float("inf")
    return max(w, h) / min(w, h)


def squareness_filter(label_map: np.ndarray, max_aspect: float) -> np.ndarray:
    """Выбрасывает (обнуляет -> background) компоненты room-классов с
    aspect ratio выше max_aspect. wall/opening/background не трогает."""
    out = label_map.copy()
    for cid in ROOM_TYPE_IDS:
        mask = (label_map == cid).astype(np.uint8)
        if not mask.any():
            continue
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        for comp_id in range(1, n):
            comp_mask = labels == comp_id
            ar = component_aspect_ratio(comp_mask)
            if ar > max_aspect:
                out[comp_mask] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--max-aspect", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
    args = ap.parse_args()

    paths = load_paths()
    ugc_test_dir = Path(paths["derived"]["ugc_test_dir"])
    output_dir = Path(paths["derived"]["output_dir"])
    gt_path = ugc_test_dir / "test_coco.json"

    coco = COCO(str(gt_path))
    room_by_img, ignore_by_img = split_ignore_regions(gt_path)
    with open(output_dir / args.model / "predictions" / "test_predictions.json", encoding="utf-8") as f:
        preds = json.load(f)
    preds = [p for p in preds if p.get("score", 1.0) >= args.score_thresh]
    img_wh = {im["id"]: (im["height"], im["width"]) for im in coco.loadImgs(coco.getImgIds())}
    preds = mask_nms(preds, img_wh, iou_thresh=args.nms_iou)

    class_ids = [3, 4, 5, 6, 7, ROOM_ID]
    names = {3: "bathroom", 4: "kitchen", 5: "balcony", 6: "wall", 7: "opening", ROOM_ID: "room"}

    variant_labels = ["baseline"] + [f"max_aspect={a}" for a in args.max_aspect]
    aggs = {v: {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids} for v in variant_labels}

    for img in coco.loadImgs(coco.getImgIds()):
        img_id, h, w = img["id"], img["height"], img["width"]
        gtm = gt_label_map(coco, img_id, h, w, room_by_img.get(img_id, []))
        pm = pred_label_map(preds, img_id, h, w, args.score_thresh)
        valid = ~true_ignore_mask(ignore_by_img, img_id, h, w)

        evaluate_dense(gtm, pm, valid, aggs["baseline"], class_ids)
        for a in args.max_aspect:
            filtered = squareness_filter(pm, max_aspect=a)
            evaluate_dense(gtm, filtered, valid, aggs[f"max_aspect={a}"], class_ids)

    print(f"{'Вариант':16s} {'IoU':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for v in variant_labels:
        o = summarize(aggs[v], names)["overall_macro"]
        print(f"{v:16s} {o['iou']:6.3f} {o['precision']:6.3f} {o['recall']:6.3f} {o['f1']:6.3f}")


if __name__ == "__main__":
    main()
