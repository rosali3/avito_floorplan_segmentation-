"""
Инференс дообученного SAM (mask_decoder из finetune_sam.py) на test-сплите.

Боксы — из ТОГО ЖЕ GroundingDINO-детектора, что и zero_shot_grounded_sam.py
(см. models/sam/grounding_boxes.py), чтобы разница в метриках между
sam_zeroshot и sam_finetuned отражала ИСКЛЮЧИТЕЛЬНО эффект дообучения
mask_decoder, а не разные источники боксов.

Запуск:
    python models/sam/infer_and_eval_finetuned.py \
        --gdino-config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --gdino-checkpoint groundingdino_swint_ogc.pth \
        --sam-checkpoint sam_vit_b_01ec64.pth --sam-model-type vit_b \
        --decoder-checkpoint output/sam_finetuned/checkpoints/best.pt
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
    ap.add_argument("--sam-checkpoint", required=True, help="исходные веса SAM (та же архитектура, что при файнтюне)")
    ap.add_argument("--sam-model-type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--decoder-checkpoint", required=True, help="output/sam_finetuned/checkpoints/best.pt")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
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
    decoder_ckpt = torch.load(args.decoder_checkpoint, map_location=args.device)
    assert decoder_ckpt["model_type"] == args.sam_model_type, (
        f"чекпоинт decoder'а обучен для {decoder_ckpt['model_type']}, а не {args.sam_model_type}"
    )
    sam.mask_decoder.load_state_dict(decoder_ckpt["mask_decoder"])
    sam.eval()
    predictor = SamPredictor(sam)
    print(f"[sam finetuned infer] decoder epoch={decoder_ckpt['epoch']} "
          f"val_loss={decoder_ckpt['val_loss']:.4f}")

    ugc_dir = Path(paths["derived"]["ugc_test_dir"])
    with open(ugc_dir / "test_coco.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    predictions = []
    for img_rec in gt["images"]:
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
                "score": det["score"] * float(iou_preds[0]),
                "segmentation": binary_mask_to_rle(mask),
            })

    out_dir = Path(paths["derived"]["output_dir"]) / "sam_finetuned" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[sam finetuned infer] {len(predictions)} predictions -> {pred_path}")

    gt_path = ugc_dir / "test_coco.json"
    for iou_type in ("segm", "bbox"):
        metrics = run_coco_eval(gt_path, predictions, iou_type=iou_type)
        with open(out_dir / f"metrics_{iou_type}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[sam finetuned infer] {iou_type}: AP@[.5:.95]={metrics['overall']['AP@[.5:.95]']:.4f} "
              f"AP@.50={metrics['overall']['AP@.50']:.4f}")


if __name__ == "__main__":
    main()
