"""
Пиксельные метрики (IoU/Dice/Precision/Recall/F1) на нашем собственном
held-out val-сплите (data/valid_coco.json, ResPlan+CubiCasa), отдельно по
источнику (resplan/cubicasa) и вместе — для RF-DETR и UNet (+ лучший
пост-процессинг UNet: Canny+dilate+connectedComponents+majority vote,
см. docs/room_postprocessing_experiments.md).

В отличие от compute_pixel_metrics.py (заточен под UGC test: ignore_regions,
room=living∪bedroom) — здесь ПРЯМАЯ разметка по всем 7 канонический классам,
никакого room-merge не нужно (GT уже различает living/bedroom нативно).

Запуск:
    python eval/compute_valid_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from mask_nms import mask_nms  # noqa: E402
from image_based_room_regions import detect_room_regions, majority_fill_by_regions  # noqa: E402

CLASS_NAMES = {1: "living", 2: "bedroom", 3: "bathroom", 4: "kitchen",
               5: "balcony", 6: "wall", 7: "opening"}
ROOM_TYPE_IDS = [1, 2, 3, 4, 5]


def _decode(seg, h, w) -> np.ndarray:
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def pred_label_map(preds_by_img_id: dict, img_id: int, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.int32)
    preds = sorted(preds_by_img_id.get(img_id, []), key=lambda p: p.get("score", 1.0))
    for p in preds:
        m = _decode(p["segmentation"], h, w)
        out[m] = p["category_id"]
    return out


def gt_label_map(coco: COCO, img_id: int, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.int32)
    anns = sorted(coco.imgToAnns.get(img_id, []), key=lambda a: -a.get("area", 0))
    for ann in anns:
        m = _decode(ann["segmentation"], h, w)
        out[m] = ann["category_id"]
    return out


def evaluate(agg: dict, gt_map: np.ndarray, pred_map: np.ndarray) -> None:
    for cid in CLASS_NAMES:
        g = gt_map == cid
        p = pred_map == cid
        agg[cid]["tp"] += int((p & g).sum())
        agg[cid]["fp"] += int((p & ~g).sum())
        agg[cid]["fn"] += int((~p & g).sum())


def summarize(agg: dict) -> dict:
    per_cat = {}
    for cid, name in CLASS_NAMES.items():
        tp, fp, fn = agg[cid]["tp"], agg[cid]["fp"], agg[cid]["fn"]
        denom_iou = tp + fp + fn
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else float("nan")
        per_cat[name] = {
            "iou": tp / denom_iou if denom_iou else float("nan"),
            "precision": p, "recall": r, "f1": f1,
        }

    def macro(key):
        vals = [v[key] for v in per_cat.values() if v[key] == v[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    return {"per_category": per_cat,
            "overall_macro": {k: macro(k) for k in ("iou", "precision", "recall", "f1")}}


def main():
    paths = load_paths()
    gt_path = Path(paths["derived"]["data_dir"]) / "valid_coco.json"
    images_root = Path(paths["combined_out_root"]) / "images"
    output_dir = Path(paths["derived"]["output_dir"])

    coco = COCO(str(gt_path))
    img_ids = coco.getImgIds()
    img_info = {i: coco.loadImgs([i])[0] for i in img_ids}
    img_wh = {i: (info["height"], info["width"]) for i, info in img_info.items()}
    source_of = {i: info["file_name"].split("/")[0] for i, info in img_info.items()}

    with open(output_dir / "rfdetr_seg_valid" / "predictions" / "valid_predictions.json", encoding="utf-8") as f:
        rfdetr_preds = json.load(f)
    rfdetr_preds = mask_nms(rfdetr_preds, img_wh, iou_thresh=0.5)
    rfdetr_by_img: dict[int, list] = {}
    for p in rfdetr_preds:
        rfdetr_by_img.setdefault(p["image_id"], []).append(p)

    with open(output_dir / "unet_baseline_valid" / "predictions" / "valid_predictions.json", encoding="utf-8") as f:
        unet_preds = json.load(f)
    unet_by_img: dict[int, list] = {}
    for p in unet_preds:
        unet_by_img.setdefault(p["image_id"], []).append(p)

    sources = ["resplan", "cubicasa", "all"]
    models = ["rfdetr_seg", "unet_baseline", "unet_baseline_cannyfill"]
    aggs = {s: {m: {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in CLASS_NAMES} for m in models} for s in sources}

    for i, img_id in enumerate(img_ids):
        info = img_info[img_id]
        h, w = info["height"], info["width"]
        src = source_of[img_id]
        gtm = gt_label_map(coco, img_id, h, w)

        rfdetr_pm = pred_label_map(rfdetr_by_img, img_id, h, w)
        unet_pm = pred_label_map(unet_by_img, img_id, h, w)

        image_bgr = cv2.imread(str(images_root / info["file_name"]))
        regions = detect_room_regions(image_bgr, 80, 200, 2, 0.01)
        unet_filled = majority_fill_by_regions(unet_pm, regions, 0.3)

        for s in (src, "all"):
            evaluate(aggs[s]["rfdetr_seg"], gtm, rfdetr_pm)
            evaluate(aggs[s]["unet_baseline"], gtm, unet_pm)
            evaluate(aggs[s]["unet_baseline_cannyfill"], gtm, unet_filled)

        if (i + 1) % 500 == 0:
            print(f"[compute_valid_metrics] {i + 1}/{len(img_ids)}")

    results = {}
    for s in sources:
        results[s] = {m: summarize(aggs[s][m]) for m in models}

    out_path = output_dir / "valid_metrics_by_source.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[compute_valid_metrics] -> {out_path}")

    for s in sources:
        print(f"\n=== source={s} ===")
        for m in models:
            o = results[s][m]["overall_macro"]
            print(f"  {m:28s} mIoU={o['iou']:.3f} P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}")
            for cname, v in results[s][m]["per_category"].items():
                print(f"      {cname:10s} iou={v['iou']:.3f} p={v['precision']:.3f} r={v['recall']:.3f} f1={v['f1']:.3f}")


if __name__ == "__main__":
    main()
