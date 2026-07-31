"""
Zero-shot instance segmentation: Grounded-SAM (GroundingDINO по текстовым
промптам классов -> боксы -> ванильный SAM без дообучения строит маски по
боксам). Веса SAM НЕ дообучаются в этом скрипте — это чистый zero-shot
бейзлайн, сравни с models/sam/finetune_sam.py + infer_and_eval_finetuned.py.

Установка (см. SETUP.md):
    pip install segment-anything pycocotools
    git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO && pip install -e .
    # скачать sam_vit_h_4b8939.pth (или vit_b/vit_l) и groundingdino_swint_ogc.pth

Запуск:
    python models/sam/zero_shot_grounded_sam.py \
        --gdino-config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --gdino-checkpoint groundingdino_swint_ogc.pth \
        --sam-checkpoint sam_vit_h_4b8939.pth --sam-model-type vit_h
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
sys.path.insert(0, str(Path(__file__).parent))
from coco_utils import load_classes, load_paths  # noqa: E402
from coco_eval_common import binary_mask_to_rle, run_coco_eval  # noqa: E402
from grounding_boxes import GroundingBoxDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdino-config", required=True)
    ap.add_argument("--gdino-checkpoint", required=True)
    ap.add_argument("--sam-checkpoint", required=True)
    ap.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-name", default="sam_zeroshot",
                     help="подпапка в output/ — не меняй, если только не гоняешь несколько вариантов порогов")
    args = ap.parse_args()

    import cv2
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    paths = load_paths()
    classes_cfg = load_classes()
    fg = classes_cfg["foreground_classes"]
    canon_names = [fg[i] for i in sorted(fg.keys(), key=int)]
    canon_name_to_id = {name: int(cid) for cid, name in fg.items()}

    detector = GroundingBoxDetector(
        config_path=args.gdino_config, checkpoint_path=args.gdino_checkpoint,
        canonical_class_names=canon_names, canon_name_to_id=canon_name_to_id,
        device=args.device, box_threshold=args.box_threshold, text_threshold=args.text_threshold,
    )

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(args.device)
    predictor = SamPredictor(sam)

    ugc_dir = Path(paths["derived"]["ugc_test_dir"])
    with open(ugc_dir / "test_coco.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    predictions = []
    for i, img_rec in enumerate(gt["images"]):
        print(f"[sam zero-shot] {i+1}/{len(gt['images'])} {img_rec['file_name']}", flush=True)
        img_path = str(ugc_dir / "images" / img_rec["file_name"])
        dets = detector.detect(img_path)
        if not dets:
            continue

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        for det in dets:
            box = np.array(det["bbox_xyxy"])
            masks, iou_preds, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0]
            x0, y0, x1, y1 = det["bbox_xyxy"]
            predictions.append({
                "image_id": img_rec["id"],
                "category_id": det["category_id"],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": det["score"] * float(iou_preds[0]),  # уверенность детекции * уверенность маски
                "segmentation": binary_mask_to_rle(mask),
            })

    out_dir = Path(paths["derived"]["output_dir"]) / args.output_name / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[sam zero-shot] {len(predictions)} predictions -> {pred_path}")

    gt_path = ugc_dir / "test_coco.json"
    for iou_type in ("segm", "bbox"):
        metrics = run_coco_eval(gt_path, predictions, iou_type=iou_type)
        with open(out_dir / f"metrics_{iou_type}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[sam zero-shot] {iou_type}: AP@[.5:.95]={metrics['overall']['AP@[.5:.95]']:.4f} "
              f"AP@.50={metrics['overall']['AP@.50']:.4f}")


if __name__ == "__main__":
    main()
