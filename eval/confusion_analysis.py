"""
Настоящая confusion matrix (не per-category AP, а "какой GT-класс с каким
предсказанным путается"): для каждого GT-инстанса ищем предсказание с
максимальным mask IoU (независимо от его класса) выше --iou-thr; если такое
нашлось — пара (GT_класс, predicted_класс) идёт в матрицу; если нет — это
false negative (пропуск). Предсказания без сматченного GT — false positive.

Запуск:
    python eval/confusion_analysis.py --gt data/ugc_test/test_coco.json \
        --pred output/rfdetr_seg/predictions/test_predictions.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


def gt_ann_to_rle(coco: COCO, ann: dict) -> dict:
    img = coco.imgs[ann["image_id"]]
    h, w = img["height"], img["width"]
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        return mask_utils.merge(rles)
    return seg  # уже RLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--score-thr", type=float, default=0.3)
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    preds = json.load(open(args.pred, "r", encoding="utf-8"))
    preds = [p for p in preds if p["score"] >= args.score_thr]
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)

    confusion = defaultdict(lambda: defaultdict(int))  # confusion[gt_class][pred_class] += 1
    fn_count = defaultdict(int)   # пропущенные GT по классу
    fp_count = defaultdict(int)   # лишние предсказания по классу (не сматчились ни с одним GT)
    matched_pred_ids = set()

    for image_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        anns = coco_gt.loadAnns(ann_ids)
        img_preds = preds_by_image.get(image_id, [])
        img = coco_gt.imgs[image_id]
        h, w = img["height"], img["width"]

        pred_rles = [p.get("segmentation") for p in img_preds]

        for ann in anns:
            gt_rle = gt_ann_to_rle(coco_gt, ann)
            best_iou, best_j = 0.0, -1
            for j, p in enumerate(img_preds):
                if id(p) in matched_pred_ids:
                    continue
                iou = mask_utils.iou([gt_rle], [pred_rles[j]], [0])[0][0]
                if iou > best_iou:
                    best_iou, best_j = iou, j
            gt_name = cat_names[ann["category_id"]]
            if best_iou >= args.iou_thr and best_j >= 0:
                pred_name = cat_names[img_preds[best_j]["category_id"]]
                confusion[gt_name][pred_name] += 1
                matched_pred_ids.add(id(img_preds[best_j]))
            else:
                fn_count[gt_name] += 1

        for p in img_preds:
            if id(p) not in matched_pred_ids:
                fp_count[cat_names[p["category_id"]]] += 1

    names = sorted(cat_names.values())
    print(f"\n=== Confusion matrix (IoU>={args.iou_thr}, score>={args.score_thr}) ===")
    print("GT \\ Pred".ljust(12), *[n[:8].rjust(9) for n in names], "  MISSED")
    for gt_name in names:
        row = [confusion[gt_name].get(p, 0) for p in names]
        print(gt_name.ljust(12), *[str(v).rjust(9) for v in row], " ", fn_count.get(gt_name, 0))

    print("\n=== False positives (предсказан класс, для которого нет подходящего GT) ===")
    for name in names:
        if fp_count.get(name):
            print(f"  {name}: {fp_count[name]}")


if __name__ == "__main__":
    main()
