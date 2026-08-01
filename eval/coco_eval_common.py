"""
Единая обёртка над pycocotools COCOeval — ей пользуются ВСЕ models/*/infer_and_eval.py,
чтобы числа между RF-DETR / YOLO-seg / Mask R-CNN / SegFormer / SAM были посчитаны
буквально одной и той же функцией на одном и том же test_coco.json (data/ugc_test/test_coco.json).

Формат предсказаний — стандартный COCO "results" JSON:
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float,
      "segmentation": {"size": [h,w], "counts": "..."} }, ...]
category_id и image_id должны соответствовать id из test_coco.json (см.
data_prep/prepare_ugc_test.py) — используй canonical category id (не 0-based idx!).

CLI:
    python eval/coco_eval_common.py --gt data/ugc_test/test_coco.json \
        --pred output/yolo_seg/predictions/test_predictions.json \
        --out output/yolo_seg/predictions/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """binary_mask: HxW uint8/bool -> COCO RLE dict (для поля segmentation в results)."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def polygon_to_rle(polygons: list[list[float]], height: int, width: int) -> dict:
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def filter_predictions_in_ignore_regions(gt_json_path: str | Path, predictions: list[dict],
                                          overlap_thresh: float = 0.5) -> tuple[list[dict], int]:
    """Выбрасывает предсказания, чей bbox на >= overlap_thresh своей площади лежит
    внутри ignore_regions (см. data_prep/prepare_ugc_test.py) — геометрия
    категорий room/coridor/hall/stairs/storage, исключённых из GT-таксономии.
    Модель не должна штрафоваться (как FP) за предсказание living/bedroom/etc.
    именно в такой зоне — истинный тип "room" нам неизвестен.

    Возвращает (отфильтрованный список предсказаний, число выброшенных).
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    ignore_regions = gt.get("ignore_regions", [])
    if not ignore_regions:
        return predictions, 0

    img_wh = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    ignore_by_img: dict[int, list[dict]] = {}
    for region in ignore_regions:
        ignore_by_img.setdefault(region["image_id"], []).append(region)

    kept = []
    n_dropped = 0
    for pred in predictions:
        img_id = pred["image_id"]
        regions = ignore_by_img.get(img_id)
        if not regions or img_id not in img_wh:
            kept.append(pred)
            continue
        h, w = img_wh[img_id]
        pred_mask = _pred_seg_to_mask(pred["segmentation"], h, w)
        pred_area = pred_mask.sum()
        if pred_area == 0:
            kept.append(pred)
            continue
        ignore_mask = np.zeros((h, w), dtype=bool)
        for region in regions:
            ignore_mask |= _pred_seg_to_mask(region["segmentation"], h, w)
        overlap = np.logical_and(pred_mask, ignore_mask).sum() / pred_area
        if overlap >= overlap_thresh:
            n_dropped += 1
        else:
            kept.append(pred)
    return kept, n_dropped


def _pred_seg_to_mask(seg, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def run_coco_eval(gt_json_path: str | Path, predictions, iou_type: str = "segm",
                   filter_ignore_regions: bool = True) -> dict:
    """predictions: путь к json ИЛИ уже загруженный list[dict] в формате COCO results.

    Возвращает dict с ключевыми метриками + per-category AP@[.5:.95] и AP50.
    """
    coco_gt = COCO(str(gt_json_path))

    if isinstance(predictions, (str, Path)):
        with open(predictions, "r", encoding="utf-8") as f:
            predictions = json.load(f)

    if filter_ignore_regions:
        predictions, n_dropped = filter_predictions_in_ignore_regions(gt_json_path, predictions)
        if n_dropped:
            print(f"[coco_eval_common] отфильтровано предсказаний в ignore_regions (room/hall/...): {n_dropped}")

    if len(predictions) == 0:
        print("[coco_eval_common] ПРЕДУПРЕЖДЕНИЕ: пустой список предсказаний — "
              "все метрики будут 0.")
        coco_dt = coco_gt.loadRes(_dummy_pred(coco_gt))
    else:
        coco_dt = coco_gt.loadRes(predictions)

    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    metrics = {
        "AP@[.5:.95]": float(ev.stats[0]),
        "AP@.50": float(ev.stats[1]),
        "AP@.75": float(ev.stats[2]),
        "AP_small": float(ev.stats[3]),
        "AP_medium": float(ev.stats[4]),
        "AP_large": float(ev.stats[5]),
        "AR@1": float(ev.stats[6]),
        "AR@10": float(ev.stats[7]),
        "AR@100": float(ev.stats[8]),
    }

    # per-category AP@[.5:.95] и AP50 (полезно для анализа слабых классов, напр. balcony)
    per_cat = {}
    cat_ids = coco_gt.getCatIds()
    cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    precisions = ev.eval["precision"]  # [T, R, K, A, M]
    for k_idx, cat_id in enumerate(cat_ids):
        name = cat_id_to_name[cat_id]
        p_all = precisions[:, :, k_idx, 0, -1]
        p_50 = precisions[0, :, k_idx, 0, -1]
        ap_all = p_all[p_all > -1].mean() if (p_all > -1).any() else float("nan")
        ap_50 = p_50[p_50 > -1].mean() if (p_50 > -1).any() else float("nan")
        per_cat[name] = {"AP@[.5:.95]": float(ap_all), "AP@.50": float(ap_50)}

    return {"iou_type": iou_type, "overall": metrics, "per_category": per_cat}


def _dummy_pred(coco_gt: COCO) -> list[dict]:
    """Не даём pycocotools упасть на пустом списке — подсовываем один
    заведомо неверный dummy-бокс, чтобы summarize() отработал и вернул нули."""
    img_ids = coco_gt.getImgIds()
    cat_ids = coco_gt.getCatIds()
    if not img_ids or not cat_ids:
        raise RuntimeError("test_coco.json пуст (нет images или categories)")
    return [{
        "image_id": img_ids[0], "category_id": cat_ids[0],
        "bbox": [0, 0, 1, 1], "score": 0.001,
        "segmentation": binary_mask_to_rle(np.zeros((1, 1), dtype=np.uint8)),
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-type", default="segm", choices=["segm", "bbox"])
    args = ap.parse_args()

    metrics = run_coco_eval(args.gt, args.pred, iou_type=args.iou_type)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[coco_eval_common] -> {args.out}")


if __name__ == "__main__":
    main()
