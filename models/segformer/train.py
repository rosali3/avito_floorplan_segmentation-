"""
Обучение SegFormer как СЕМАНТИЧЕСКОГО сегментатора (это единственная модель
в исследовании, которая по своей природе не даёт instances "из коробки" —
instances для сравнения с остальными моделями получаются постфактум,
connected-components по предсказанной семантической карте, см. infer_and_eval.py).

Датасет и аугментации — БЕЗ ДУБЛИРОВАНИЯ переиспользуем существующий, уже
проверенный `combined_out/resplan_dataset.py` (импортируем как есть, не правим,
как и просит README_GENERIC.md), но со СВОИМ 80/20 сплитом (train_files.txt /
valid_files.txt из data_prep/build_train_val_coco.py) через параметр `ids=`,
который в ResPlanSegmentation как раз предназначен "переопределить split".

Известный баг аугментаций (WALL_ID/OPENING_IDS от старой 12-классовой
таксономии, см. combined_out/README_GENERIC.md) действует и здесь, как и на
все модели, обученные этим датасетом — сознательно не чиним, чтобы не давать
SegFormer нечестное преимущество по классам wall/opening.

Перед запуском:
    pip install torch torchvision transformers accelerate albumentations opencv-python-headless tensorboard
    python data_prep/build_train_val_coco.py   # создаёт data/train_files.txt, data/valid_files.txt

Запуск (из корня claude_instseg_compare/):
    python models/segformer/train.py
    python models/segformer/train.py --epochs 60 --batch-size 8
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md
# ВАЖНО: должно стоять раньше "import torch" ниже — иначе torch уже
# инициализирует CUDA context со всеми видимыми GPU.

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import mlflow
except ImportError:
    mlflow = None  # опционально: pip install mlflow, иначе просто CSV+TensorBoard

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
from coco_utils import load_paths  # noqa: E402

NUM_CLASSES = 8  # background(0) + 7 foreground, см. configs/classes.yaml
IGNORE_INDEX = 255


class SegFormerWrap(nn.Module):
    """Контракт: вход [B,3,H,W] -> логиты [B,NUM_CLASSES,H,W] (без softmax/argmax)."""

    def __init__(self, num_classes: int, pretrained: str = "nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        logits = self.net(pixel_values=x).logits  # [B,C,H/4,W/4]
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def load_median_class_weights(combined_out_root: Path, num_classes: int) -> torch.Tensor:
    """Median-frequency balancing из pixel_frequency.json — тот же протокол,
    что train_generic.py использует для остальных моделей в репозитории."""
    with open(combined_out_root / "pixel_frequency.json", "r", encoding="utf-8") as f:
        freq = json.load(f)
    names = ["background", "living", "bedroom", "bathroom", "kitchen", "balcony", "wall", "opening"]
    shares = np.array([freq["share"][n] for n in names], dtype=np.float64)
    median = np.median(shares)
    weights = median / shares
    return torch.tensor(weights, dtype=torch.float32)[:num_classes]


def compute_miou(conf_mat: np.ndarray) -> tuple[float, np.ndarray]:
    ious = np.zeros(conf_mat.shape[0])
    for c in range(conf_mat.shape[0]):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        denom = tp + fp + fn
        ious[c] = tp / denom if denom > 0 else float("nan")
    return float(np.nanmean(ious)), ious


def update_conf_mat(conf_mat: np.ndarray, pred: np.ndarray, target: np.ndarray, num_classes: int):
    mask = target != IGNORE_INDEX
    p = pred[mask]
    t = target[mask]
    idx = t.astype(np.int64) * num_classes + p.astype(np.int64)
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    conf_mat += binc.reshape(num_classes, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--workers", type=int, default=0, help="0 рекомендовано на Windows")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true", help="выключить MLflow-логирование")
    args = ap.parse_args()

    paths = load_paths()
    combined_out_root = Path(paths["combined_out_root"])
    data_dir = Path(paths["derived"]["data_dir"])
    out_dir = Path(paths["derived"]["output_dir"]) / "segformer"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(combined_out_root))
    from resplan_dataset import ResPlanSegmentation  # noqa: E402  (импортируем как есть, не редактируем)

    def read_ids(p: Path) -> list[str]:
        return [ln.strip() for ln in open(p, "r", encoding="utf-8") if ln.strip()]

    train_ids = read_ids(data_dir / "train_files.txt")
    valid_ids = read_ids(data_dir / "valid_files.txt")

    train_ds = ResPlanSegmentation(root=str(combined_out_root), split="train", variant="train", ids=train_ids)
    valid_ds = ResPlanSegmentation(root=str(combined_out_root), split="val", variant="val", ids=valid_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers)
    print(f"[segformer train] train={len(train_ds)} valid={len(valid_ds)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegFormerWrap(NUM_CLASSES).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = load_median_class_weights(combined_out_root, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    writer = SummaryWriter(log_dir=str(log_dir))
    csv_path = log_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "lr", "seconds"])

    use_mlflow = mlflow is not None and not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(f"file:{(out_dir / 'mlruns').as_posix()}")
        mlflow.set_experiment("segformer")
        mlflow.start_run()
        mlflow.log_params({
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "class_weights": not args.no_class_weights, "num_classes": NUM_CLASSES,
            "train_size": len(train_ds), "valid_size": len(valid_ds),
        })
        print(f"[segformer] MLflow: file:{(out_dir / 'mlruns').as_posix()} "
              f"(смотреть: mlflow ui --backend-store-uri file:{(out_dir / 'mlruns').as_posix()})")

    best_miou = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(1, n_batches)

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, masks)
                val_loss_sum += loss.item()
                n_val_batches += 1
                pred = logits.argmax(1).cpu().numpy()
                update_conf_mat(conf_mat, pred, masks.cpu().numpy(), NUM_CLASSES)
        val_loss = val_loss_sum / max(1, n_val_batches)
        val_miou, per_class_iou = compute_miou(conf_mat)
        scheduler.step(val_miou)
        dt = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[segformer] epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} lr={lr_now:.2e} {dt:.1f}s")
        csv_writer.writerow([epoch, train_loss, val_loss, val_miou, lr_now, round(dt, 1)])
        csv_file.flush()
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metrics/val_mIoU", val_miou, epoch)
        if use_mlflow:
            mlflow.log_metrics(
                {"loss/train": train_loss, "loss/val": val_loss, "metrics/val_mIoU": val_miou, "lr": lr_now},
                step=epoch,
            )

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": val_miou},
                   ckpt_dir / "final.pt")
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": val_miou},
                       ckpt_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[segformer] early stopping: {args.patience} эпох без улучшения val_mIoU")
                break

    csv_file.close()
    writer.close()
    if use_mlflow:
        mlflow.log_metric("best_val_mIoU", best_miou)
        mlflow.log_artifact(str(ckpt_dir / "best.pt"))
        mlflow.end_run()
    print(f"[segformer train] done. best_val_mIoU={best_miou:.4f}. "
          f"CSV -> {csv_path}, TensorBoard -> {log_dir}, чекпоинты -> {ckpt_dir}")


if __name__ == "__main__":
    main()
