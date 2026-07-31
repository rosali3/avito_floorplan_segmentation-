"""
Собирает output/<model>/predictions/metrics_{segm,bbox}.json всех моделей в
одну итоговую таблицу (CSV + Markdown).

Ожидаемая раскладка (создаётся каждым models/*/infer_and_eval.py):
    output/<model_name>/predictions/test_predictions.json   (сырые COCO results)
    output/<model_name>/predictions/metrics_segm.json       (run_coco_eval iou_type=segm)
    output/<model_name>/predictions/metrics_bbox.json       (run_coco_eval iou_type=bbox)

Запуск (после того, как отработали все infer_and_eval.py):
    python eval/aggregate_results.py
Результат:
    output/final_results.csv
    output/final_results.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402

MODEL_DIRS = [
    ("rfdetr_seg", "RF-DETR-Seg (Medium)"),
    ("yolo_seg", "YOLO-seg (ultralytics)"),
    ("maskrcnn_mmdet", "Mask R-CNN (MMDetection)"),
    ("segformer", "SegFormer (+connected components)"),
    ("sam_zeroshot", "SAM zero-shot (Grounded-SAM)"),
    ("sam_finetuned", "SAM fine-tuned"),
]


def load_metrics(output_dir: Path, model_key: str) -> dict | None:
    seg_path = output_dir / model_key / "predictions" / "metrics_segm.json"
    box_path = output_dir / model_key / "predictions" / "metrics_bbox.json"
    if not seg_path.is_file() or not box_path.is_file():
        return None
    with open(seg_path, "r", encoding="utf-8") as f:
        seg = json.load(f)
    with open(box_path, "r", encoding="utf-8") as f:
        box = json.load(f)
    return {"segm": seg, "bbox": box}


def main():
    paths = load_paths()
    output_dir = Path(paths["derived"]["output_dir"])

    rows = []
    per_category_rows = []
    for model_key, model_label in MODEL_DIRS:
        m = load_metrics(output_dir, model_key)
        if m is None:
            rows.append({
                "model": model_label, "status": "NOT RUN / no metrics found",
                "mAP50_bbox": "", "mAP50-95_bbox": "",
                "mAP50_mask": "", "mAP50-95_mask": "", "AR100_mask": "",
            })
            continue
        seg_o = m["segm"]["overall"]
        box_o = m["bbox"]["overall"]
        rows.append({
            "model": model_label, "status": "ok",
            "mAP50_bbox": round(box_o["AP@.50"], 4),
            "mAP50-95_bbox": round(box_o["AP@[.5:.95]"], 4),
            "mAP50_mask": round(seg_o["AP@.50"], 4),
            "mAP50-95_mask": round(seg_o["AP@[.5:.95]"], 4),
            "AR100_mask": round(seg_o["AR@100"], 4),
        })
        for cat_name, cat_metrics in m["segm"]["per_category"].items():
            per_category_rows.append({
                "model": model_label, "category": cat_name,
                "mask_AP50": round(cat_metrics["AP@.50"], 4) if cat_metrics["AP@.50"] == cat_metrics["AP@.50"] else "",
                "mask_AP50-95": round(cat_metrics["AP@[.5:.95]"], 4) if cat_metrics["AP@[.5:.95]"] == cat_metrics["AP@[.5:.95]"] else "",
            })

    out_csv = output_dir / "final_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_cat_csv = output_dir / "final_results_per_category.csv"
    if per_category_rows:
        with open(out_cat_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_category_rows[0].keys()))
            w.writeheader()
            w.writerows(per_category_rows)

    out_md = output_dir / "final_results.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Итоговое сравнение моделей instance segmentation "
                "(test = ugc_labeled, объединённый train+valid+test)\n\n")
        f.write("| Модель | mAP50 (box) | mAP50-95 (box) | mAP50 (mask) | mAP50-95 (mask) | AR100 (mask) | статус |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['mAP50_bbox']} | {r['mAP50-95_bbox']} | "
                    f"{r['mAP50_mask']} | {r['mAP50-95_mask']} | {r['AR100_mask']} | {r['status']} |\n")
        if per_category_rows:
            f.write("\n## Per-category mask AP\n\n")
            f.write("| Модель | Класс | mask AP50 | mask AP50-95 |\n|---|---|---:|---:|\n")
            for r in per_category_rows:
                f.write(f"| {r['model']} | {r['category']} | {r['mask_AP50']} | {r['mask_AP50-95']} |\n")

    print(f"[aggregate_results] -> {out_csv}\n[aggregate_results] -> {out_md}")
    for r in rows:
        print(f"  {r['model']:35s} {r['status']}")


if __name__ == "__main__":
    main()
