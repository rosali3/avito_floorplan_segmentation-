"""
Инференс RF-DETR / UNet-baseline на нашем собственном held-out val-сплите
(data/valid_coco.json, ResPlan+CubiCasa, тот же 80/20 seed=42, что
использовался при обучении RF-DETR/YOLO/MaskRCNN/SegFormer) — в отличие от
всех остальных eval-скриптов в проекте, которые всегда работают на UGC
test. Нужно, чтобы сравнить синтетическую валидацию с реальным UGC отдельно
по источникам (resplan vs cubicasa).

ВАЖНО: UNet обучался ВНЕ этого проекта на своём собственном train/val
сплите (другой код, ResPlan-main) — этот val-сплит для UNet НЕ
гарантированно honest holdout (могло пересечься с его тренировочными
данными).

Запуск:
    python eval/run_valid_inference.py --model rfdetr_seg
    python eval/run_valid_inference.py --model unet_baseline
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # локальная машина: единственный GPU = индекс 0;
# models/unet_baseline/infer_and_eval.py делает setdefault(...,"1") для удалённого
# мульти-GPU сервера — при импорте на этой машине это указывает на несуществующий
# GPU и роняет процесс segfault'ом, если не выставить явно ДО импорта.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from coco_eval_common import binary_mask_to_rle  # noqa: E402


def run_rfdetr(gt_path: Path, images_root: Path, paths: dict, threshold: float,
                model_key: str = "rfdetr_seg") -> list[dict]:
    from rfdetr import RFDETRSegMedium

    if model_key == "rfdetr_seg_fullaug":
        # fullaug обучен на data_v2 (ремаплено в наши 7 канонических классов
        # через data_prep/remap_fullaug_v2.py) — тот же порядок категорий,
        # но своя копия train_coco.json, не data/rfdetr_ds
        train_ann = Path(paths["project_root"]) / "data_v2" / "train_coco.json"
    else:
        train_ann = Path(paths["derived"]["rfdetr_dataset_dir"]) / "train" / "_annotations.coco.json"
    with open(train_ann, "r", encoding="utf-8") as f:
        train_coco = json.load(f)
    cats = sorted(train_coco["categories"], key=lambda c: c["id"])
    class_id_map = {i: c["id"] for i, c in enumerate(cats)}

    ckpt = Path(paths["derived"]["output_dir"]) / model_key / "checkpoints" / "checkpoint_best_ema.pth"
    model = RFDETRSegMedium(pretrain_weights=str(ckpt))

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    predictions = []
    for i, img_rec in enumerate(gt["images"]):
        image = Image.open(images_root / img_rec["file_name"]).convert("RGB")
        dets = model.predict(image, threshold=threshold)
        n = len(dets.xyxy)
        has_mask = getattr(dets, "mask", None) is not None
        for j in range(n):
            cid_0based = int(dets.class_id[j])
            if cid_0based not in class_id_map:
                # редкий edge-case: чекпоинт помечен как 7-классовый, но голова
                # модели по факту сохраняет больше слотов (см. warning про
                # "Checkpoint has 7 classes but model is configured for 90") —
                # изредка вылезает индекс вне наших 7 категорий, пропускаем
                continue
            x0, y0, x1, y1 = dets.xyxy[j].tolist()
            pred = {
                "image_id": img_rec["id"],
                "category_id": class_id_map[cid_0based],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(dets.confidence[j]),
            }
            if has_mask:
                pred["segmentation"] = binary_mask_to_rle(np.asarray(dets.mask[j]))
            else:
                mask = np.zeros((img_rec["height"], img_rec["width"]), dtype=np.uint8)
                xi0, yi0, xi1, yi1 = map(int, [x0, y0, x1, y1])
                mask[max(0, yi0):yi1, max(0, xi0):xi1] = 1
                pred["segmentation"] = binary_mask_to_rle(mask)
            predictions.append(pred)
        if (i + 1) % 500 == 0:
            print(f"[rfdetr valid infer] {i + 1}/{len(gt['images'])}")
    return predictions


def run_unet(gt_path: Path, images_root: Path, paths: dict) -> list[dict]:
    import cv2
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "unet_baseline"))
    from infer_and_eval import UNet, preprocess  # noqa: E402
    from coco_utils import load_classes, mask_to_instances  # noqa: E402
    from coco_eval_common import polygon_to_rle  # noqa: E402

    classes_cfg = load_classes()
    fg_ids = sorted(int(k) for k in classes_cfg["foreground_classes"].keys())

    ckpt_path = Path(paths["derived"]["output_dir"]) / "unet_baseline_best_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = UNet(num_classes=ckpt["num_classes"], base=ckpt["base_filters"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[unet valid infer] loaded epoch={ckpt.get('epoch')} val_iou={ckpt.get('val_iou')}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    predictions = []
    with torch.no_grad():
        for i, img_rec in enumerate(gt["images"]):
            image_bgr = cv2.imread(str(images_root / img_rec["file_name"]), cv2.IMREAD_COLOR)
            h, w = img_rec["height"], img_rec["width"]
            x = preprocess(image_bgr).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_small = probs.argmax(0).astype(np.uint8)
            pred_mask = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)
            prob_of_pred = cv2.resize(probs.max(0), (w, h), interpolation=cv2.INTER_LINEAR)

            instances = mask_to_instances(
                pred_mask, foreground_class_ids=fg_ids,
                min_area_px=classes_cfg["min_instance_area_px"],
                approx_epsilon_frac=classes_cfg["polygon_approx_epsilon_frac"],
            )
            for inst in instances:
                inst_mask = np.zeros((h, w), dtype=np.uint8)
                for poly in inst["segmentation"]:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(inst_mask, [pts], 1)
                score = float(prob_of_pred[inst_mask.astype(bool)].mean()) if inst_mask.sum() else 0.01
                predictions.append({
                    "image_id": img_rec["id"],
                    "category_id": inst["category_id"],
                    "bbox": inst["bbox"],
                    "score": score,
                    "segmentation": polygon_to_rle(inst["segmentation"], h, w),
                })
            if (i + 1) % 500 == 0:
                print(f"[unet valid infer] {i + 1}/{len(gt['images'])}")
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     choices=["rfdetr_seg", "rfdetr_seg_fullaug", "unet_baseline"])
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    paths = load_paths()
    gt_path = Path(paths["derived"]["data_dir"]) / "valid_coco.json"
    images_root = Path(paths["combined_out_root"]) / "images"

    if args.model in ("rfdetr_seg", "rfdetr_seg_fullaug"):
        predictions = run_rfdetr(gt_path, images_root, paths, args.threshold, model_key=args.model)
    else:
        predictions = run_unet(gt_path, images_root, paths)

    out_dir = Path(paths["derived"]["output_dir"]) / f"{args.model}_valid" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "valid_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[run_valid_inference] {len(predictions)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
