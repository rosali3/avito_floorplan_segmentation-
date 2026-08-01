"""
Пиксельные метрики (IoU, Dice, Precision, Recall, Accuracy) поверх тех же
test_predictions.json / GT coco json, которыми уже пользуется coco_eval_common.py
для COCO mAP. Даёт единую метрику, сравнимую между instance-моделями
(RF-DETR/YOLO/Mask R-CNN/SAM) и semantic-моделями (SegFormer/UNet), т.к. обе
группы сводятся к бинарным маскам класса на изображении (для семантических
моделей это и так их родной выход, для instance — объединение всех
предсказанных/GT-инстансов класса).

Для каждого класса и картинки:
    P = объединение (union) масок всех предсказаний класса с score >= thresh
    G = объединение (union) масок всех GT-инстансов класса
    TP = |P & G|, FP = |P \\ G|, FN = |G \\ P|, TN = остальные пиксели картинки
Агрегация — микро (суммируем TP/FP/FN/TN по всем картинкам класса), т.к. многие
картинки не содержат данный класс вообще (macro по картинкам был бы нестабилен
из-за деления на ноль/вырожденных случаев).

Запуск:
    python eval/compute_pixel_metrics.py \\
        --gt data/ugc_test/test_coco.json \\
        --pred output/yolo_seg/predictions/test_predictions.json \\
        --out output/yolo_seg/predictions/pixel_metrics.json \\
        --score-thresh 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


def _ann_to_binary_mask(coco_gt: COCO, ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, dict):  # RLE
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)


def _pred_to_binary_mask(pred: dict, h: int, w: int) -> np.ndarray:
    seg = pred["segmentation"]
    if isinstance(seg, dict):  # RLE (наш стандартный формат в test_predictions.json)
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)


def compute_pixel_metrics(gt_json_path: str | Path, predictions, score_thresh: float = 0.3) -> dict:
    coco_gt = COCO(str(gt_json_path))
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    ignore_by_img: dict[int, list[dict]] = {}
    for region in gt_raw.get("ignore_regions", []):
        ignore_by_img.setdefault(region["image_id"], []).append(region)

    if isinstance(predictions, (str, Path)):
        with open(predictions, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    predictions = [p for p in predictions if p.get("score", 1.0) >= score_thresh]

    preds_by_img_cat: dict[tuple[int, int], list[dict]] = {}
    for p in predictions:
        preds_by_img_cat.setdefault((p["image_id"], p["category_id"]), []).append(p)

    gt_by_img_cat: dict[tuple[int, int], list[dict]] = {}
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        gt_by_img_cat.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)

    cat_ids = coco_gt.getCatIds()
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    img_info = {im["id"]: im for im in coco_gt.loadImgs(coco_gt.getImgIds())}

    # tp/fp/fn/tn в пикселях, суммарно по всем картинкам, отдельно на класс
    agg = {cid: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cid in cat_ids}

    for img_id, info in img_info.items():
        h, w = info["height"], info["width"]

        valid_mask = np.ones((h, w), dtype=bool)
        for region in ignore_by_img.get(img_id, []):
            valid_mask &= ~_ann_to_binary_mask(coco_gt, region, h, w)

        for cid in cat_ids:
            gt_anns = gt_by_img_cat.get((img_id, cid), [])
            pred_anns = preds_by_img_cat.get((img_id, cid), [])
            if not gt_anns and not pred_anns:
                continue  # класса нет ни в GT, ни в предсказаниях на этой картинке — пропускаем, не раздуваем TN нулями

            g_mask = np.zeros((h, w), dtype=bool)
            for ann in gt_anns:
                g_mask |= _ann_to_binary_mask(coco_gt, ann, h, w)

            p_mask = np.zeros((h, w), dtype=bool)
            for pred in pred_anns:
                p_mask |= _pred_to_binary_mask(pred, h, w)

            # пиксели внутри ignore_regions (room/coridor/hall/...) не считаем ни
            # в чью пользу — истинный класс там неизвестен, нельзя судить о TP/FP
            g_mask &= valid_mask
            p_mask &= valid_mask

            tp = int(np.logical_and(p_mask, g_mask).sum())
            fp = int(np.logical_and(p_mask, ~g_mask).sum())
            fn = int(np.logical_and(~p_mask, g_mask).sum())
            tn = int(np.logical_and(np.logical_and(~p_mask, ~g_mask), valid_mask).sum())
            agg[cid]["tp"] += tp
            agg[cid]["fp"] += fp
            agg[cid]["fn"] += fn
            agg[cid]["tn"] += tn

    per_category = {}
    for cid in cat_ids:
        a = agg[cid]
        tp, fp, fn, tn = a["tp"], a["fp"], a["fn"], a["tn"]
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        denom_prec = tp + fp
        denom_rec = tp + fn
        denom_acc = tp + fp + fn + tn
        per_category[cat_id_to_name[cid]] = {
            "iou": tp / denom_iou if denom_iou else float("nan"),
            "dice": 2 * tp / denom_dice if denom_dice else float("nan"),
            "precision": tp / denom_prec if denom_prec else float("nan"),
            "recall": tp / denom_rec if denom_rec else float("nan"),
            "accuracy": (tp + tn) / denom_acc if denom_acc else float("nan"),
            "n_images_with_gt_or_pred": sum(
                1 for img_id in img_info
                if gt_by_img_cat.get((img_id, cid)) or preds_by_img_cat.get((img_id, cid))
            ),
        }

    def macro(key: str) -> float:
        vals = [v[key] for v in per_category.values() if v[key] == v[key]]  # drop NaN
        return sum(vals) / len(vals) if vals else float("nan")

    overall = {k: macro(k) for k in ("iou", "dice", "precision", "recall", "accuracy")}
    return {"score_thresh": score_thresh, "overall_macro": overall, "per_category": per_category}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    args = ap.parse_args()

    metrics = compute_pixel_metrics(args.gt, args.pred, score_thresh=args.score_thresh)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[compute_pixel_metrics] -> {args.out}")
    print(f"[compute_pixel_metrics] overall (macro over classes): {metrics['overall_macro']}")


if __name__ == "__main__":
    main()
