"""
Инференс обученного YOLO-seg на test-сплите (data/ugc_test/) + подсчёт метрик
через общий eval/coco_eval_common.py.

class index (0-based, ultralytics) -> canonical category_id восстанавливаем
той же сортировкой foreground_classes по id, что использовал
data_prep/coco_to_yolo_seg.py при генерации data.yaml (никакого отдельного
файла-маппинга не нужно, порядок детерминирован конфигом).

Запуск:
    python models/yolo_seg/infer_and_eval.py --weights output/yolo_seg/checkpoints/best.pt
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from coco_utils import load_classes, load_paths  # noqa: E402
from coco_eval_common import polygon_to_rle, run_coco_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model-key", default="yolo_seg",
                     help="папка в output/ для предсказаний+метрик — задай отдельное имя "
                          "(напр. yolo_seg_fullaug), чтобы не перезаписать результаты другого чекпоинта")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    classes_cfg = load_classes()
    fg = classes_cfg["foreground_classes"]
    ordered_ids = sorted(int(k) for k in fg.keys())
    idx_to_canon_id = {i: cid for i, cid in enumerate(ordered_ids)}

    ugc_dir = Path(paths["derived"]["ugc_test_dir"])
    with open(ugc_dir / "test_coco.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    model = YOLO(args.weights)

    predictions = []
    for img_rec in gt["images"]:
        img_path = ugc_dir / "images" / img_rec["file_name"]
        results = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        r = results[0]
        if r.masks is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        polys_xy = r.masks.xy  # список массивов Nx2 в координатах исходного изображения

        for i in range(len(boxes)):
            x0, y0, x1, y1 = boxes[i].tolist()
            poly = polys_xy[i]
            if poly is None or len(poly) < 3:
                continue
            flat_poly = poly.flatten().tolist()
            rle = polygon_to_rle([flat_poly], img_rec["height"], img_rec["width"])
            predictions.append({
                "image_id": img_rec["id"],
                "category_id": idx_to_canon_id[int(clss[i])],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(confs[i]),
                "segmentation": rle,
            })

    out_dir = Path(paths["derived"]["output_dir"]) / args.model_key / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[yolo infer] {len(predictions)} predictions -> {pred_path}")

    gt_path = ugc_dir / "test_coco.json"
    for iou_type in ("segm", "bbox"):
        metrics = run_coco_eval(gt_path, predictions, iou_type=iou_type)
        with open(out_dir / f"metrics_{iou_type}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[yolo infer] {iou_type}: AP@[.5:.95]={metrics['overall']['AP@[.5:.95]']:.4f} "
              f"AP@.50={metrics['overall']['AP@.50']:.4f}")


if __name__ == "__main__":
    main()
