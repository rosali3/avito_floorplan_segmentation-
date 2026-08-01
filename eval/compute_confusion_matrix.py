"""
Пиксельная confusion matrix (8x8: background + 7 классов) на UGC test.

Каждому пикселю картинки (кроме ignore_regions — room/hall/...) присваивается
ОДИН GT-класс и ОДИН предсказанный класс (если несколько масок перекрываются —
для GT побеждает класс с меньшей площадью инстанса, для предсказаний — с
большим score; это тот же принцип, что и в data_prep/make_ugc_semantic_masks.py).
Предсказания фильтруются по единому --score-thresh — ЧТОБЫ старые и новые
(fullaug) версии моделей сравнивались на одинаковом пороге уверенности.

Запуск (одна модель):
    python eval/compute_confusion_matrix.py --model rfdetr_seg --score-thresh 0.3
Запуск (сравнение старая/новая для RF-DETR и YOLO при одном threshold):
    python eval/compute_confusion_matrix.py --compare-fullaug --score-thresh 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402

CLASS_NAMES = ["background", "living", "bedroom", "bathroom", "kitchen", "balcony", "wall", "opening"]


def seg_to_mask(seg, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def gt_label_map(coco: COCO, img_id: int, h: int, w: int) -> np.ndarray:
    """0=background, 1-7=класс; при перекрытии побеждает МЕНЬШИЙ инстанс (рисуется последним)."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    anns_sorted = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)
    label = np.zeros((h, w), dtype=np.uint8)
    for ann in anns_sorted:
        label[seg_to_mask(ann["segmentation"], h, w)] = ann["category_id"]
    return label


def pred_label_map(preds: list[dict], img_id: int, h: int, w: int, score_thresh: float) -> np.ndarray:
    """0=фон/нет предсказания, 1-7=класс; при перекрытии побеждает БОЛЬШИЙ score (рисуется последним)."""
    relevant = [p for p in preds if p["image_id"] == img_id and p.get("score", 1.0) >= score_thresh]
    relevant.sort(key=lambda p: p.get("score", 1.0))  # по возрастанию -> самый уверенный красится последним
    label = np.zeros((h, w), dtype=np.uint8)
    for p in relevant:
        label[seg_to_mask(p["segmentation"], h, w)] = p["category_id"]
    return label


def ignore_mask(ignore_by_img: dict[int, list[dict]], img_id: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for region in ignore_by_img.get(img_id, []):
        m |= seg_to_mask(region["segmentation"], h, w)
    return m


def compute_confusion(gt_json_path: Path, pred_json_path: Path, score_thresh: float) -> np.ndarray:
    coco = COCO(str(gt_json_path))
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    ignore_by_img: dict[int, list[dict]] = {}
    for region in gt_raw.get("ignore_regions", []):
        ignore_by_img.setdefault(region["image_id"], []).append(region)

    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=np.int64)

    for img in coco.loadImgs(coco.getImgIds()):
        img_id, h, w = img["id"], img["height"], img["width"]
        gt_map = gt_label_map(coco, img_id, h, w)
        pred_map = pred_label_map(preds, img_id, h, w, score_thresh)
        valid = ~ignore_mask(ignore_by_img, img_id, h, w)

        gt_flat = gt_map[valid]
        pred_flat = pred_map[valid]
        idx = gt_flat.astype(np.int64) * n + pred_flat.astype(np.int64)
        counts = np.bincount(idx, minlength=n * n)
        cm += counts.reshape(n, n)

    return cm


def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("GT (реальность)")
    ax.set_title(title)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="доля от GT-класса (по строке)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[compute_confusion_matrix] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="ключ модели (папка в output/), напр. rfdetr_seg")
    ap.add_argument("--compare-fullaug", action="store_true",
                     help="посчитать rfdetr_seg vs rfdetr_seg_fullaug и yolo_seg vs yolo_seg_fullaug")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--out-dir", default="docs/report_assets/confusion_matrices")
    args = ap.parse_args()

    paths = load_paths()
    ugc_test_dir = Path(paths["derived"]["ugc_test_dir"])
    output_dir = Path(paths["derived"]["output_dir"])
    gt_path = ugc_test_dir / "test_coco.json"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_fullaug:
        models = ["rfdetr_seg", "rfdetr_seg_fullaug", "yolo_seg", "yolo_seg_fullaug"]
    elif args.model:
        models = [args.model]
    else:
        raise SystemExit("укажи --model <ключ> или --compare-fullaug")

    for model_key in models:
        pred_path = output_dir / model_key / "predictions" / "test_predictions.json"
        if not pred_path.is_file():
            print(f"[compute_confusion_matrix] пропуск {model_key}: нет {pred_path}")
            continue
        cm = compute_confusion(gt_path, pred_path, args.score_thresh)
        np.savetxt(out_dir / f"{model_key}_confusion.csv", cm, fmt="%d", delimiter=",",
                    header=",".join(CLASS_NAMES), comments="")
        plot_confusion(cm, f"{model_key} (score>={args.score_thresh})", out_dir / f"{model_key}_confusion.png")


if __name__ == "__main__":
    main()
