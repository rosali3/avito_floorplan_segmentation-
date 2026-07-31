"""
"Мягкая" оценка поверх СЫРЫХ ugc_labeled аннотаций (не отфильтрованных):
- GT "room" считается правильно найденным, если предсказанный класс — living
  ИЛИ bedroom (но НЕ kitchen — это по-прежнему считаем содержательной ошибкой,
  см. progress: RF-DETR предсказывал room->kitchen в 41% случаев).
- GT "bathroom"/"restroom" считаются правильными при предсказании "bathroom"
  (в официальном test_coco.json restroom УЖЕ смёржен в bathroom на этапе
  prepare_ugc_test.py — тут просто явно подтверждаем это на сырых данных).
- hall/coridor/stairs/storage — по-прежнему вне рассмотрения (нет разумного
  соответствия в нашей таксономии, см. classes.yaml).
- Остальные классы (kitchen/balcony/wall/opening) — как раньше, точное совпадение.

Это НЕ пересчитанный mAP (что потребовало бы честной precision-recall кривой
с "OR"-классами, что COCOeval не поддерживает из коробки) — это recall-style
сводка "сколько GT-инстансов нашли/пропустили" при мягких правилах, для прямого
sравнения со строгим results.

Запуск:
    python eval/lenient_check.py --ugc-labeled-root "C:/.../ugc_labeled" \
        --pred output/rfdetr_seg/predictions/test_predictions.json \
        --merged-gt data/ugc_test/test_coco.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as mask_utils

SPLITS = ("train", "valid", "test")

# GT-класс (сырой UGC) -> множество ПРИЕМЛЕМЫХ предсказанных классов
LENIENT_ACCEPT = {
    "room": {"living", "bedroom"},
    "bathroom": {"bathroom"},
    "restroom": {"bathroom"},
    "kitchen": {"kitchen"},
    "balcony": {"balcony"},
    "wall": {"wall"},
    "door": {"opening"},
    "window": {"opening"},
    # enterence больше НЕ считаем эквивалентом opening — по запросу сливаем
    # его с coridor в отдельную группу (см. IGNORED, оба вне рассмотрения тут же)
}
IGNORED = {"hall", "coridor", "stairs", "storage", "toilet", "enterence"}
MERGE_GROUPS = {"coridor": "coridor+enterence", "enterence": "coridor+enterence"}


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
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--score-thr", type=float, default=0.3)
    args = ap.parse_args()

    ugc_root = Path(args.ugc_labeled_root)
    merged_gt = json.load(open(args.merged_gt, "r", encoding="utf-8"))
    merged_name_to_id = {img["file_name"]: img["id"] for img in merged_gt["images"]}
    merged_cat_names = {c["id"]: c["name"] for c in merged_gt["categories"]}

    preds = json.load(open(args.pred, "r", encoding="utf-8"))
    preds = [p for p in preds if p["score"] >= args.score_thr]
    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)

    matched = defaultdict(int)
    missed = defaultdict(int)
    total = defaultdict(int)
    ignored_count = defaultdict(int)

    for split in SPLITS:
        ann_path = ugc_root / split / "_annotations.coco.json"
        if not ann_path.is_file():
            continue
        raw = json.load(open(ann_path, "r", encoding="utf-8"))
        raw_cat_names = {c["id"]: c["name"] for c in raw["categories"]}
        raw_img_by_id = {im["id"]: im for im in raw["images"]}

        for ann in raw["annotations"]:
            raw_cat = raw_cat_names[ann["category_id"]]
            if raw_cat in IGNORED:
                ignored_count[MERGE_GROUPS.get(raw_cat, raw_cat)] += 1
                continue
            if raw_cat not in LENIENT_ACCEPT:
                continue
            img = raw_img_by_id[ann["image_id"]]
            merged_name = f"{split}__{img['file_name']}"
            merged_id = merged_name_to_id.get(merged_name)
            if merged_id is None:
                continue

            h, w = img["height"], img["width"]
            gt_rle = poly_to_rle(ann["segmentation"], h, w)
            accept_set = LENIENT_ACCEPT[raw_cat]

            best_iou, best_ok = 0.0, False
            for p in preds_by_image.get(merged_id, []):
                iou = mask_utils.iou([gt_rle], [p["segmentation"]], [0])[0][0]
                if iou >= args.iou_thr and iou > best_iou:
                    best_iou = iou
                    best_ok = merged_cat_names[p["category_id"]] in accept_set

            total[raw_cat] += 1
            if best_iou >= args.iou_thr and best_ok:
                matched[raw_cat] += 1
            else:
                missed[raw_cat] += 1

    print(f"=== Мягкий recall (IoU>={args.iou_thr}, score>={args.score_thr}) ===\n")
    print(f"{'класс':10s} {'найдено':>8s} {'пропущено':>10s} {'recall':>8s}  приемлемые предсказанные классы")
    for cat, acc in LENIENT_ACCEPT.items():
        t = total.get(cat, 0)
        if t == 0:
            continue
        m = matched.get(cat, 0)
        print(f"{cat:10s} {m:8d} {missed.get(cat,0):10d} {m/t:8.1%}  {sorted(acc)}")
    print(f"\n(вне рассмотрения, как и раньше: {dict(ignored_count)})")


if __name__ == "__main__":
    main()
