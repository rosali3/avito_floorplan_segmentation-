"""
Инференс сторонней UNet-simple (обученной ВНЕ этого проекта, см. avito-toilet/
ResPlan-main/train_resplan_v2.py, чекпоинт /home/User24/checkpoints_combined/
best_model.pt на школьном сервере) на test-сплите (data/ugc_test/) + перевод
СЕМАНТИЧЕСКОЙ карты в instances (тот же connected-components код, что и для
SegFormer, см. data_prep/coco_utils.mask_to_instances) + метрики через общий
eval/coco_eval_common.py.

Класс-схема чекпоинта (id_to_name в combined_out/class_mapping.json) СОВПАДАЕТ
1:1 с нашей канонической таксономией (0=background, 1..7=living..opening) —
поэтому маппинг тривиальный, доп. merge не нужен.

Архитектура и предобработка скопированы из ResPlan-main/eval_combined.py и
resplan_dataset.py (ImageNet mean/std, resize 256x256) — НЕ редактируй тот
внешний код, здесь только независимая копия для инференса.

Запуск (на school1, в /home/User24/venv):
    python models/unet_baseline/infer_and_eval.py \
        --checkpoint /home/User24/checkpoints_combined/best_model.pt
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from coco_utils import load_classes, load_paths, mask_to_instances  # noqa: E402
from coco_eval_common import polygon_to_rle, run_coco_eval  # noqa: E402

TRAIN_RESOLUTION = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


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
    ap.add_argument("--output-name", default="unet_baseline")
    args = ap.parse_args()

    paths = load_paths()
    classes_cfg = load_classes()
    fg_ids = sorted(int(k) for k in classes_cfg["foreground_classes"].keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet(num_classes=ckpt["num_classes"], base=ckpt["base_filters"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[unet infer] loaded checkpoint epoch={ckpt.get('epoch')} "
          f"val_iou={ckpt.get('val_iou')} num_classes={ckpt['num_classes']}")

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

    out_dir = Path(paths["derived"]["output_dir"]) / args.output_name / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[unet infer] {len(predictions)} predictions -> {pred_path}")

    gt_path = ugc_dir / "test_coco.json"
    for iou_type in ("segm", "bbox"):
        metrics = run_coco_eval(gt_path, predictions, iou_type=iou_type)
        with open(out_dir / f"metrics_{iou_type}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[unet infer] {iou_type}: AP@[.5:.95]={metrics['overall']['AP@[.5:.95]']:.4f} "
              f"AP@.50={metrics['overall']['AP@.50']:.4f}")


if __name__ == "__main__":
    main()
