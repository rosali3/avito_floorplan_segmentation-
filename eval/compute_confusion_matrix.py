"""
Пиксельная confusion matrix на UGC test: строки GT (9: background + 7 классов +
room), столбцы — предсказания модели КАК ЕСТЬ (8: background + 7 классов, без
искусственного "room" — модель никогда не предсказывает room, это категория
только сырой UGC-разметки).

Раз сырая UGC-разметка вообще не различает living/bedroom (там только общая
категория "room"), для строки GT="room" правильным ответом считается ЛЮБОЕ из
living/bedroom — такие ячейки подсвечиваются жирным зелёным текстом ("ok").
Предсказания при этом НЕ переименовываются и не объединяются — если модель
предсказала living там, где GT="bathroom", это по-прежнему видно как реальная
ошибка (GT=bathroom, предсказано=living), а не маскируется под room.

coridor/hall/stairs/storage/toilet — по-прежнему невозможно сопоставить ни с
чем осмысленным, остаются полностью исключены (true ignore).

Каждому пикселю картинки присваивается ОДИН GT-класс и ОДИН предсказанный
класс (при перекрытии для GT побеждает класс с меньшей площадью инстанса,
для предсказаний — с большим score).

Запуск (одна модель):
    python eval/compute_confusion_matrix.py --model rfdetr_seg --score-thresh 0.3
Запуск (все модели, отдельный порог для RF-DETR/YOLO):
    python eval/compute_confusion_matrix.py --all --score-thresh 0.3 --rfdetr-thresh 0.1 --yolo-thresh 0.1
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

MODEL_KEYS = [
    "rfdetr_seg", "rfdetr_seg_fullaug", "yolo_seg", "yolo_seg_fullaug",
    "maskrcnn_mmdet", "segformer", "sam_zeroshot", "sam_finetuned", "unet_baseline",
]

ROW_CLASS_NAMES = ["background", "living", "bedroom", "bathroom", "kitchen", "balcony", "wall", "opening", "room"]
COL_CLASS_NAMES = ["background", "living", "bedroom", "bathroom", "kitchen", "balcony", "wall", "opening"]
ROOM_ID = 8
LIVING_ID, BEDROOM_ID = 1, 2
ROOM_ROW_IDX = ROW_CLASS_NAMES.index("room")
LIVING_COL_IDX = COL_CLASS_NAMES.index("living")
BEDROOM_COL_IDX = COL_CLASS_NAMES.index("bedroom")


def seg_to_mask(seg, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def gt_label_map(coco: COCO, img_id: int, h: int, w: int, room_regions: list[dict]) -> np.ndarray:
    """0=background, 1-7=класс, 8=room; сначала кладём room (низкий приоритет),
    потом реальные аннотации по убыванию площади (мелкие перекрывают крупные)."""
    label = np.zeros((h, w), dtype=np.uint8)
    for region in room_regions:
        label[seg_to_mask(region["segmentation"], h, w)] = ROOM_ID

    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    anns_sorted = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)
    for ann in anns_sorted:
        label[seg_to_mask(ann["segmentation"], h, w)] = ann["category_id"]
    return label


def pred_label_map(preds: list[dict], img_id: int, h: int, w: int, score_thresh: float) -> np.ndarray:
    """0=фон/нет предсказания, 1-7=класс — РОВНО то, что предсказала модель, без remap."""
    relevant = [p for p in preds if p["image_id"] == img_id and p.get("score", 1.0) >= score_thresh]
    relevant.sort(key=lambda p: p.get("score", 1.0))  # по возрастанию -> самый уверенный красится последним
    label = np.zeros((h, w), dtype=np.uint8)
    for p in relevant:
        label[seg_to_mask(p["segmentation"], h, w)] = p["category_id"]
    return label


def split_ignore_regions(gt_json_path: Path) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """Возвращает (room_regions_by_img, true_ignore_by_img) — room сравнивается,
    coridor/hall/stairs/storage/toilet по-прежнему полностью исключаются."""
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    room_by_img: dict[int, list[dict]] = {}
    ignore_by_img: dict[int, list[dict]] = {}
    for region in gt_raw.get("ignore_regions", []):
        bucket = room_by_img if region.get("raw_name") == "room" else ignore_by_img
        bucket.setdefault(region["image_id"], []).append(region)
    return room_by_img, ignore_by_img


def true_ignore_mask(ignore_by_img: dict[int, list[dict]], img_id: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for region in ignore_by_img.get(img_id, []):
        m |= seg_to_mask(region["segmentation"], h, w)
    return m


def compute_confusion(gt_json_path: Path, pred_json_path: Path, score_thresh: float) -> np.ndarray:
    coco = COCO(str(gt_json_path))
    room_by_img, ignore_by_img = split_ignore_regions(gt_json_path)

    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    n_rows, n_cols = len(ROW_CLASS_NAMES), len(COL_CLASS_NAMES)
    cm = np.zeros((n_rows, n_cols), dtype=np.int64)

    for img in coco.loadImgs(coco.getImgIds()):
        img_id, h, w = img["id"], img["height"], img["width"]
        gt_map = gt_label_map(coco, img_id, h, w, room_by_img.get(img_id, []))
        pred_map = pred_label_map(preds, img_id, h, w, score_thresh)
        valid = ~true_ignore_mask(ignore_by_img, img_id, h, w)

        gt_flat = gt_map[valid]
        pred_flat = pred_map[valid]
        idx = gt_flat.astype(np.int64) * n_cols + pred_flat.astype(np.int64)
        counts = np.bincount(idx, minlength=n_rows * n_cols)
        cm += counts.reshape(n_rows, n_cols)

    return cm


def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COL_CLASS_NAMES)))
    ax.set_yticks(range(len(ROW_CLASS_NAMES)))
    ax.set_xticklabels(COL_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(ROW_CLASS_NAMES)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("GT (реальность)")
    ax.set_title(title)
    for i in range(len(ROW_CLASS_NAMES)):
        for j in range(len(COL_CLASS_NAMES)):
            if cm[i, j] == 0:
                continue
            is_room_ok = (i == ROOM_ROW_IDX) and (j in (LIVING_COL_IDX, BEDROOM_COL_IDX))
            label = f"{norm[i, j]:.2f}" + (" ok" if is_room_ok else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                     fontweight="bold" if is_room_ok else "normal",
                     color="limegreen" if is_room_ok else ("white" if norm[i, j] > 0.5 else "black"))
    fig.colorbar(im, ax=ax, label="доля от GT-класса (по строке)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[compute_confusion_matrix] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="ключ модели (папка в output/), напр. rfdetr_seg")
    ap.add_argument("--all", action="store_true", help="посчитать для всех доступных моделей")
    ap.add_argument("--score-thresh", type=float, default=0.1,
                     help="единый порог confidence для ВСЕХ моделей (честное сравнение)")
    ap.add_argument("--rfdetr-thresh", type=float, default=None,
                     help="опционально: отдельный порог для rfdetr_seg/rfdetr_seg_fullaug")
    ap.add_argument("--yolo-thresh", type=float, default=None,
                     help="опционально: отдельный порог для yolo_seg/yolo_seg_fullaug")
    ap.add_argument("--out-dir", default="docs/report_assets/confusion_matrices")
    args = ap.parse_args()

    paths = load_paths()
    ugc_test_dir = Path(paths["derived"]["ugc_test_dir"])
    output_dir = Path(paths["derived"]["output_dir"])
    gt_path = ugc_test_dir / "test_coco.json"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        models = MODEL_KEYS
    elif args.model:
        models = [args.model]
    else:
        raise SystemExit("укажи --model <ключ> или --all")

    thresh_by_model = {}
    if args.rfdetr_thresh is not None:
        thresh_by_model["rfdetr_seg"] = args.rfdetr_thresh
        thresh_by_model["rfdetr_seg_fullaug"] = args.rfdetr_thresh
    if args.yolo_thresh is not None:
        thresh_by_model["yolo_seg"] = args.yolo_thresh
        thresh_by_model["yolo_seg_fullaug"] = args.yolo_thresh

    for model_key in models:
        pred_path = output_dir / model_key / "predictions" / "test_predictions.json"
        if not pred_path.is_file():
            print(f"[compute_confusion_matrix] пропуск {model_key}: нет {pred_path}")
            continue
        thresh = thresh_by_model.get(model_key, args.score_thresh)
        cm = compute_confusion(gt_path, pred_path, thresh)
        np.savetxt(out_dir / f"{model_key}_confusion.csv", cm, fmt="%d", delimiter=",",
                    header=",".join(COL_CLASS_NAMES), comments="")
        plot_confusion(cm, f"{model_key} (score>={thresh})", out_dir / f"{model_key}_confusion.png")


if __name__ == "__main__":
    main()
