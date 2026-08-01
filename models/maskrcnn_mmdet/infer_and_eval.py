"""
Инференс обученного Mask R-CNN на test-сплите (data/ugc_test/) + метрики через
общий eval/coco_eval_common.py.

label (0-based) из mmdet соответствует порядку CLASS_NAMES в config_maskrcnn.py
(отсортирован по возрастанию canonical id) -> восстанавливаем canonical
category_id той же сортировкой, что и в config_maskrcnn.py/coco_to_yolo_seg.py.

Запуск (в venv с mmdetection):
    python models/maskrcnn_mmdet/infer_and_eval.py --checkpoint output/maskrcnn_mmdet/checkpoints/best.pth
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from coco_utils import load_classes, load_paths  # noqa: E402
from coco_eval_common import binary_mask_to_rle, run_coco_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--model-key", default="maskrcnn_mmdet",
                     help="папка в output/ для предсказаний+метрик — задай отдельное имя, "
                          "чтобы не перезаписать результаты другого чекпоинта")
    args = ap.parse_args()

    from mmdet.apis import init_detector, inference_detector

    cfg_path = Path(__file__).parent / "config_maskrcnn.py"
    model = init_detector(str(cfg_path), args.checkpoint, device="cuda:0")

    paths = load_paths()
    classes_cfg = load_classes()
    fg = classes_cfg["foreground_classes"]
    ordered_ids = sorted(int(k) for k in fg.keys())  # тот же порядок, что CLASS_NAMES в конфиге
    idx_to_canon_id = {i: cid for i, cid in enumerate(ordered_ids)}

    ugc_dir = Path(paths["derived"]["ugc_test_dir"])
    with open(ugc_dir / "test_coco.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    predictions = []
    for img_rec in gt["images"]:
        img_path = str(ugc_dir / "images" / img_rec["file_name"])
        result = inference_detector(model, img_path)
        inst = result.pred_instances
        bboxes = inst.bboxes.cpu().numpy()
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()
        masks = inst.masks.cpu().numpy() if hasattr(inst, "masks") else None

        for i in range(len(bboxes)):
            if scores[i] < args.score_thr:
                continue
            x0, y0, x1, y1 = bboxes[i].tolist()
            pred = {
                "image_id": img_rec["id"],
                "category_id": idx_to_canon_id[int(labels[i])],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(scores[i]),
            }
            if masks is not None:
                pred["segmentation"] = binary_mask_to_rle(masks[i].astype(np.uint8))
            predictions.append(pred)

    out_dir = Path(paths["derived"]["output_dir"]) / args.model_key / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[maskrcnn infer] {len(predictions)} predictions -> {pred_path}")

    gt_path = ugc_dir / "test_coco.json"
    for iou_type in ("segm", "bbox"):
        metrics = run_coco_eval(gt_path, predictions, iou_type=iou_type)
        with open(out_dir / f"metrics_{iou_type}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[maskrcnn infer] {iou_type}: AP@[.5:.95]={metrics['overall']['AP@[.5:.95]']:.4f} "
              f"AP@.50={metrics['overall']['AP@.50']:.4f}")


if __name__ == "__main__":
    main()
