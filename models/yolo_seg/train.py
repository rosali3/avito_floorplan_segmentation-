"""
Обучение YOLO-seg (по умолчанию YOLO11m-seg, легко переключить на YOLOv8-seg)
через ultralytics.

Перед запуском:
    pip install ultralytics
    python data_prep/build_train_val_coco.py
    python data_prep/coco_to_yolo_seg.py

Запуск (из корня claude_instseg_compare/):
    python models/yolo_seg/train.py
    # переключить архитектуру:
    python models/yolo_seg/train.py --model yolov8m-seg.pt
    # на нашей локальной RTX 3080 8GB при OOM уменьши batch и/или imgsz:
    python models/yolo_seg/train.py --batch 4 --imgsz 512

ultralytics сам пишет per-epoch метрики в
output/yolo_seg/checkpoints/yolo_seg_run/results.csv и поддерживает TensorBoard
(--plots + `tensorboard --logdir output/yolo_seg/checkpoints`), а также сам
сохраняет лучший/последний чекпоинт в .../weights/{best,last}.pt.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
from coco_utils import load_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m-seg.pt",
                     help="базовые веса ultralytics; альтернатива: yolov8m-seg.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=30, help="early stopping")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = load_paths()
    data_yaml = Path(paths["derived"]["yolo_dataset_dir"]) / "data.yaml"
    output_dir = Path(paths["derived"]["output_dir"]) / "yolo_seg" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        project=str(output_dir),
        name="yolo_seg_run",
        exist_ok=True,
        plots=True,
        seed=42,
    )

    run_dir = output_dir / "yolo_seg_run"
    weights_dir = run_dir / "weights"
    if (weights_dir / "best.pt").is_file():
        shutil.copyfile(weights_dir / "best.pt", output_dir / "best.pt")
    if (weights_dir / "last.pt").is_file():
        shutil.copyfile(weights_dir / "last.pt", output_dir / "final.pt")

    print(f"[yolo train] done. results.csv -> {run_dir / 'results.csv'}")
    print(f"[yolo train] best -> {output_dir / 'best.pt'}, final -> {output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
