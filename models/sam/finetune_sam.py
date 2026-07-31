"""
Дообучение SAM: замораживаем image_encoder и prompt_encoder, обучаем только
mask_decoder на GT-боксах (bbox из train_coco.json) -> GT-маска (BCE + Dice).
Это стандартный, наименее ресурсоёмкий рецепт файнтюна SAM (полный файнтюн
ViT-энкодера на 8GB VRAM и ~16k картинок нереалистичен).

На инференсе (models/sam/infer_and_eval_finetuned.py) боксы для этого
чекпоинта берутся из ТОГО ЖЕ GroundingDINO-детектора, что и в zero-shot
варианте (models/sam/grounding_boxes.py) — не GT! — чтобы сравнение
"zero-shot vs finetuned" изолировало именно эффект дообучения decoder'а,
а не разницу "GT box vs detected box".

Установка: pip install segment-anything pycocotools tensorboard
Веса-старт: sam_vit_b_01ec64.pth (vit_b — компромисс качество/VRAM для 8GB карты)

Запуск (из корня claude_instseg_compare/):
    python models/sam/finetune_sam.py --sam-checkpoint sam_vit_b_01ec64.pth --sam-model-type vit_b
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md
# ВАЖНО: должно стоять раньше "import torch" ниже — иначе torch уже
# инициализирует CUDA context со всеми видимыми GPU.

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

try:
    import mlflow
except ImportError:
    mlflow = None  # опционально: pip install mlflow, иначе просто CSV+TensorBoard

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
from coco_utils import load_paths  # noqa: E402


def rasterize(polygons: list[list[float]], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(m, [pts], 1)
    return m


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1 - (2 * inter + 1) / (union + 1)


def group_by_image(coco: dict):
    by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        by_img.setdefault(ann["image_id"], []).append(ann)
    img_by_id = {im["id"]: im for im in coco["images"]}
    return by_img, img_by_id


def forward_loss(sam, transform, images_root: Path, img_rec: dict, anns: list[dict],
                  device: str, train_mode: bool) -> torch.Tensor | None:
    if not anns:
        return None
    image_bgr = cv2.imread(str(images_root / img_rec["file_name"]))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rec["height"], img_rec["width"]

    input_image = transform.apply_image(image_rgb)
    input_tensor = torch.as_tensor(input_image, device=device).permute(2, 0, 1).contiguous()[None]
    input_tensor = sam.preprocess(input_tensor)

    with torch.no_grad():
        image_embedding = sam.image_encoder(input_tensor)

    boxes_xywh = np.array([ann["bbox"] for ann in anns], dtype=np.float32)
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 2] += boxes_xyxy[:, 0]
    boxes_xyxy[:, 3] += boxes_xyxy[:, 1]
    boxes_t = transform.apply_boxes(boxes_xyxy, (h, w))
    boxes_t = torch.as_tensor(boxes_t, dtype=torch.float32, device=device)

    with torch.no_grad():
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=boxes_t, masks=None)

    low_res_masks, _iou_preds = sam.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    masks_up = sam.postprocess_masks(low_res_masks, input_tensor.shape[-2:], (h, w))[:, 0]  # [N,h,w]

    gt_masks = np.stack([rasterize(ann["segmentation"], h, w) for ann in anns])
    gt_t = torch.as_tensor(gt_masks, dtype=torch.float32, device=device)

    loss = F.binary_cross_entropy_with_logits(masks_up, gt_t) + dice_loss(masks_up, gt_t).mean()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam-checkpoint", required=True)
    ap.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-instances-per-image", type=int, default=8,
                     help="ограничение под VRAM — все instance одной картинки идут одним forward")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-dir", default=None,
                     help="папка с train_coco.json/valid_coco.json — по умолчанию "
                          "paths.yaml:derived.data_dir (укажи data_v2 для исправленных данных)")
    ap.add_argument("--no-mlflow", action="store_true", help="выключить MLflow-логирование")
    args = ap.parse_args()

    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    paths = load_paths()
    data_dir = Path(args.data_dir) if args.data_dir else Path(paths["derived"]["data_dir"])
    print(f"[sam finetune] data_dir = {data_dir}")
    images_root = Path(paths["combined_out_root"]) / "images"
    out_dir = Path(paths["derived"]["output_dir"]) / "sam_finetuned"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "train_coco.json", "r", encoding="utf-8") as f:
        train_coco = json.load(f)
    with open(data_dir / "valid_coco.json", "r", encoding="utf-8") as f:
        valid_coco = json.load(f)

    train_by_img, train_img_by_id = group_by_image(train_coco)
    valid_by_img, valid_img_by_id = group_by_image(valid_coco)

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(args.device)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr)
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    writer = SummaryWriter(log_dir=str(log_dir))
    csv_path = log_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "seconds"])

    use_mlflow = mlflow is not None and not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(f"file:{(out_dir / 'mlruns').as_posix()}")
        mlflow.set_experiment("sam_finetune")
        mlflow.start_run()
        mlflow.log_params({
            "epochs": args.epochs, "lr": args.lr, "sam_model_type": args.sam_model_type,
            "max_instances_per_image": args.max_instances_per_image,
            "train_images": len(train_by_img), "valid_images": len(valid_by_img),
        })
        print(f"[sam finetune] MLflow: file:{(out_dir / 'mlruns').as_posix()}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        sam.mask_decoder.train()
        train_ids = list(train_by_img.keys())
        random.Random(epoch).shuffle(train_ids)
        train_loss_sum, n_train = 0.0, 0
        for image_id in train_ids:
            anns = train_by_img[image_id][: args.max_instances_per_image]
            loss = forward_loss(sam, transform, images_root, train_img_by_id[image_id],
                                 anns, args.device, train_mode=True)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_train += 1
        train_loss = train_loss_sum / max(1, n_train)

        sam.mask_decoder.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for image_id in valid_by_img:
                anns = valid_by_img[image_id][: args.max_instances_per_image]
                loss = forward_loss(sam, transform, images_root, valid_img_by_id[image_id],
                                     anns, args.device, train_mode=False)
                if loss is None:
                    continue
                val_loss_sum += loss.item()
                n_val += 1
        val_loss = val_loss_sum / max(1, n_val)
        dt = time.time() - t0

        print(f"[sam finetune] epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} {dt:.1f}s")
        csv_writer.writerow([epoch, train_loss, val_loss, round(dt, 1)])
        csv_file.flush()
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        if use_mlflow:
            mlflow.log_metrics({"loss/train": train_loss, "loss/val": val_loss}, step=epoch)

        torch.save({"model_type": args.sam_model_type,
                    "mask_decoder": sam.mask_decoder.state_dict(),
                    "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "final.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_type": args.sam_model_type,
                        "mask_decoder": sam.mask_decoder.state_dict(),
                        "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "best.pt")

    csv_file.close()
    writer.close()
    if use_mlflow:
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(str(ckpt_dir / "best.pt"))
        mlflow.end_run()
    print(f"[sam finetune] done. best_val_loss={best_val_loss:.4f}. "
          f"CSV -> {csv_path}, TensorBoard -> {log_dir}, чекпоинты -> {ckpt_dir}")
    print("[sam finetune] чекпоинт хранит только mask_decoder.state_dict() — "
          "при инференсе (infer_and_eval_finetuned.py) загружается поверх того же "
          "--sam-checkpoint/--sam-model-type (image_encoder/prompt_encoder не менялись).")


if __name__ == "__main__":
    main()
