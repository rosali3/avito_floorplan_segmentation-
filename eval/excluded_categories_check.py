"""
Проверяет, ЧТО наша модель предсказывает поверх ИСКЛЮЧЁННЫХ из GT категорий
UGC (room, hall, coridor, stairs, storage — см. configs/classes.yaml
ugc_excluded_categories). Эти категории не участвуют в официальном score
(нет соответствия в train-таксономии), но полезно посмотреть постфактум,
не путает ли модель, например, "room" с "bathroom" систематически.

Работает с СЫРЫМИ ugc_labeled/{train,valid,test}/_annotations.coco.json
(до фильтрации prepare_ugc_test.py), сопоставляя по имени файла с уже
посчитанными предсказаниями (test_predictions.json).

Запуск:
    python eval/excluded_categories_check.py \
        --ugc-labeled-root "C:/Users/user/Downloads/avito-toilet/ugc_labeled" \
        --pred output/rfdetr_seg/predictions/test_predictions.json \
        --merged-gt data/ugc_test/test_coco.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as mask_utils

EXCLUDED = ["room", "hall", "coridor", "stairs", "storage", "toilet", "enterence"]
# enterence в официальной таксономии смёржен в opening (см. classes.yaml), но по
# запросу здесь его сливаем с coridor в отдельную диагностическую псевдо-категорию —
# оба похожи на переходное/входное пространство, а не на дверной проём как объект
MERGE_GROUPS = {"coridor": "coridor+enterence", "enterence": "coridor+enterence"}
SPLITS = ("train", "valid", "test")


def poly_to_rle(seg, h, w):
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        return mask_utils.merge(rles)
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ugc-labeled-root", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--merged-gt", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--score-thr", type=float, default=0.3)
    args = ap.parse_args()

    ugc_root = Path(args.ugc_labeled_root)
    merged_gt = json.load(open(args.merged_gt, "r", encoding="utf-8"))
    merged_name_to_id = {img["file_name"]: img["id"] for img in merged_gt["images"]}

    preds = json.load(open(args.pred, "r", encoding="utf-8"))
    preds = [p for p in preds if p["score"] >= args.score_thr]
    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)
    cat_names = {c["id"]: c["name"] for c in merged_gt["categories"]}

    confusion = defaultdict(lambda: defaultdict(int))
    total_by_cat = defaultdict(int)

    for split in SPLITS:
        ann_path = ugc_root / split / "_annotations.coco.json"
        if not ann_path.is_file():
            continue
        raw = json.load(open(ann_path, "r", encoding="utf-8"))
        raw_cat_names = {c["id"]: c["name"] for c in raw["categories"]}
        raw_img_by_id = {im["id"]: im for im in raw["images"]}

        for ann in raw["annotations"]:
            raw_cat_name = raw_cat_names[ann["category_id"]]
            if raw_cat_name not in EXCLUDED:
                continue
            raw_cat_name = MERGE_GROUPS.get(raw_cat_name, raw_cat_name)
            img = raw_img_by_id[ann["image_id"]]
            merged_name = f"{split}__{img['file_name']}"
            merged_id = merged_name_to_id.get(merged_name)
            if merged_id is None:
                continue

            h, w = img["height"], img["width"]
            gt_rle = poly_to_rle(ann["segmentation"], h, w)

            best_iou, best_pred = 0.0, None
            for p in preds_by_image.get(merged_id, []):
                iou = mask_utils.iou([gt_rle], [p["segmentation"]], [0])[0][0]
                if iou > best_iou:
                    best_iou, best_pred = iou, p

            total_by_cat[raw_cat_name] += 1
            if best_iou >= args.iou_thr and best_pred is not None:
                confusion[raw_cat_name][cat_names[best_pred["category_id"]]] += 1
            else:
                confusion[raw_cat_name]["(ничего не предсказано)"] += 1

    print(f"=== Что модель предсказывает поверх ИСКЛЮЧЁННЫХ из GT категорий UGC (IoU>={args.iou_thr}) ===\n")
    for cat, total in total_by_cat.items():
        print(f"{cat} (всего {total} инстансов в UGC):")
        for pred_name, cnt in sorted(confusion[cat].items(), key=lambda kv: -kv[1]):
            print(f"    -> {pred_name}: {cnt} ({100*cnt/total:.0f}%)")
        print()


if __name__ == "__main__":
    main()
