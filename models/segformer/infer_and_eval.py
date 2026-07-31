"""
Инференс SegFormer на test-сплите (data/ugc_test/) + перевод СЕМАНТИЧЕСКОЙ
карты в instances (connected components, тот же код что для построения
train/valid из масок combined_out, см. data_prep/coco_utils.mask_to_instances)
+ подсчёт метрик через общий eval/coco_eval_common.py.

Это единственная модель в исследовании без нативных instances — сравнение с
остальными честное в смысле "тот же test-сплит, та же метрика COCOeval", но
методологически SegFormer здесь оценивается в невыгодном для себя режиме
(semantic -> pseudo-instances), это отдельно проговорено в отчёте (docs/).

score каждого instance = средняя softmax-вероятность предсказанного класса по
его пикселям (у семантической сегментации нет отдельного per-instance score,
это наименее произвольная замена, дающая COCOeval осмысленную precision-recall
кривую вместо одинаковых score=1.0 у всех).

Запуск:
    python models/segformer/infer_and_eval.py --checkpoint output/segformer/checkpoints/best.pt
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md
# ВАЖНО: должно стоять раньше "import torch" ниже (и раньше "from train import ...",
# который тоже тянет torch) — иначе torch уже инициализирует CUDA context со
# всеми видимыми GPU.

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from coco_utils import load_classes, load_paths, mask_to_instances  # noqa: E402
from coco_eval_common import polygon_to_rle, run_coco_eval  # noqa: E402
from train import SegFormerWrap, NUM_CLASSES  # noqa: E402

TRAIN_RESOLUTION = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_bgr, (TRAIN_RESOLUTION, TRAIN_RESOLUTION), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def score_from_polygons(polygons: list[list[float]], prob_map: np.ndarray, height: int, width: int) -> float:
    inst_mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(inst_mask, [pts], 1)
    if inst_mask.sum() == 0:
        return 0.01
    return float(prob_map[inst_mask.astype(bool)].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    paths = load_paths()
    classes_cfg = load_classes()
    fg_ids = sorted(int(k) for k in classes_cfg["foreground_classes"].keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegFormerWrap(NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[segformer infer] loaded checkpoint epoch={ckpt.get('epoch')} "
          f"val_miou={ckpt.get('val_miou')}")

    ugc_dir = Path(paths["derived"]["ugc_test_dir"])
    with open(ugc_dir / "test_coco.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    predictions = []
    with torch.no_grad():
        for img_rec in gt["images"]:
            img_path = ugc_dir / "images" / img_rec["file_name"]
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            h, w = img_rec["height"], img_rec["width"]

            x = preprocess(image_bgr).to(device)
            logits = model(x)  # [1,C,256,256]
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # [C,256,256]
            pred_small = probs.argmax(0).astype(np.uint8)

            pred_mask = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)
            prob_of_pred_small = probs.max(0)
            prob_of_pred = cv2.resize(prob_of_pred_small, (w, h), interpolation=cv2.INTER_LINEAR)

            instances = mask_to_instances(
                pred_mask, foreground_class_ids=fg_ids,
                min_area_px=classes_cfg["min_instance_area_px"],
                approx_epsilon_frac=classes_cfg["polygon_approx_epsilon_frac"],
            )
            for inst in instances:
                score = score_from_polygons(inst["segmentation"], prob_of_pred, h, w)
                rle = polygon_to_rle(inst["segmentation"], h, w)
                predictions.append({
                    "image_id": img_rec["id"],
                    "category_id": inst["category_id"],
                    "bbox": inst["bbox"],
                    "score": score,
                    "segmentation": rle,
                })

    out_dir = Path(paths["derived"]["output_dir"]) / "segformer" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[segformer infer] {len(predictions)} predictions -> {pred_path}")

    gt_path = ugc_dir / "test_coco.json"
    for iou_type in ("segm", "bbox"):
        metrics = run_coco_eval(gt_path, predictions, iou_type=iou_type)
        with open(out_dir / f"metrics_{iou_type}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[segformer infer] {iou_type}: AP@[.5:.95]={metrics['overall']['AP@[.5:.95]']:.4f} "
              f"AP@.50={metrics['overall']['AP@.50']:.4f}")


if __name__ == "__main__":
    main()
