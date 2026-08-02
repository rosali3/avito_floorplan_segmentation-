"""
Другой источник "границ", чем предсказанная моделью wall-маска (которая
дырявая и ненадёжная) — классическое CV прямо на исходном фото плана.
Линии стен на печатном плане обычно чёткие тёмные линии, детектируются
Canny-эджами гораздо надёжнее, чем через семантическую сегментацию.

Пайплайн:
1. Canny по исходному фото (+ Gaussian blur, чтобы не ловить мелкий шум).
2. Dilate эджей — заращиваем мелкие разрывы линии (пунктир, JPEG-артефакты).
3. connectedComponents по инвертированной маске = кандидаты "комнаты".
4. Компоненты меньше --min-area-frac от площади картинки выбрасываем как шум
   (текст, мелкие значки, кусочки размерных линий) — НЕ трогаем предсказание
   там (оставляем как было).
5. Для оставшихся регионов — majority vote по предсказанному классу модели
   внутри региона (та же логика, что в wall_bounded_fill.py), заливка.

Запуск:
    python eval/image_based_room_regions.py --model segformer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from compute_confusion_matrix import (  # noqa: E402
    gt_label_map, pred_label_map, split_ignore_regions, true_ignore_mask, ROOM_ID,
)
from wall_bounded_fill import evaluate_dense, summarize, ROOM_TYPE_IDS  # noqa: E402
from mask_nms import mask_nms  # noqa: E402


def detect_room_regions(image_bgr: np.ndarray, canny_lo: int = 50, canny_hi: int = 150,
                         dilate_iters: int = 2, min_area_frac: float = 0.01,
                         use_clahe: bool = False, clahe_clip: float = 2.0) -> np.ndarray:
    """Возвращает HxW int32: 0 = граница/шум/слишком мелкая область, 1..N = room-region id."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iters)

    interior = (edges == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(interior, connectivity=8)

    min_area = min_area_frac * h * w
    out = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for comp_id in range(1, n):
        comp_mask = labels == comp_id
        if comp_mask.sum() < min_area:
            continue
        out[comp_mask] = next_id
        next_id += 1
    return out


def majority_fill_by_regions(pred_label_map_: np.ndarray, region_map: np.ndarray,
                              min_room_frac: float = 0.3) -> np.ndarray:
    out = pred_label_map_.copy()
    for region_id in range(1, region_map.max() + 1):
        m = region_map == region_id
        vals = pred_label_map_[m]
        room_vals = vals[np.isin(vals, ROOM_TYPE_IDS)]
        if len(room_vals) == 0:
            continue
        counts = np.bincount(room_vals, minlength=8)
        majority = int(counts.argmax())
        if counts[majority] / len(vals) >= min_room_frac:
            out[m] = majority
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-room-frac", type=float, default=0.3)
    ap.add_argument("--canny-lo", type=int, default=50)
    ap.add_argument("--canny-hi", type=int, default=150)
    ap.add_argument("--dilate-iters", type=int, default=2)
    ap.add_argument("--min-area-frac", type=float, default=0.01)
    ap.add_argument("--use-clahe", action="store_true")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    args = ap.parse_args()

    paths = load_paths()
    ugc_test_dir = Path(paths["derived"]["ugc_test_dir"])
    output_dir = Path(paths["derived"]["output_dir"])
    gt_path = ugc_test_dir / "test_coco.json"

    coco = COCO(str(gt_path))
    room_by_img, ignore_by_img = split_ignore_regions(gt_path)
    with open(output_dir / args.model / "predictions" / "test_predictions.json", encoding="utf-8") as f:
        preds = json.load(f)
    preds = [p for p in preds if p.get("score", 1.0) >= args.score_thresh]
    img_wh = {im["id"]: (im["height"], im["width"]) for im in coco.loadImgs(coco.getImgIds())}
    preds = mask_nms(preds, img_wh, iou_thresh=args.nms_iou)

    class_ids = [3, 4, 5, 6, 7, ROOM_ID]
    names = {3: "bathroom", 4: "kitchen", 5: "balcony", 6: "wall", 7: "opening", ROOM_ID: "room"}
    agg_before = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids}
    agg_after = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids}

    for img in coco.loadImgs(coco.getImgIds()):
        img_id, h, w = img["id"], img["height"], img["width"]
        image_bgr = cv2.imread(str(ugc_test_dir / "images" / img["file_name"]))
        if image_bgr is None:
            continue
        gtm = gt_label_map(coco, img_id, h, w, room_by_img.get(img_id, []))
        pm = pred_label_map(preds, img_id, h, w, args.score_thresh)
        valid = ~true_ignore_mask(ignore_by_img, img_id, h, w)

        evaluate_dense(gtm, pm, valid, agg_before, class_ids)

        regions = detect_room_regions(image_bgr, args.canny_lo, args.canny_hi,
                                       args.dilate_iters, args.min_area_frac,
                                       use_clahe=args.use_clahe, clahe_clip=args.clahe_clip)
        filled = majority_fill_by_regions(pm, regions, args.min_room_frac)
        evaluate_dense(gtm, filled, valid, agg_after, class_ids)

    before = summarize(agg_before, names)
    after = summarize(agg_after, names)
    print("=== ДО (сырая dense-карта) ===")
    print(before["overall_macro"])
    for c, v in before["per_category"].items():
        print(f"  {c:10s} f1={v['f1']:.3f}")
    print("=== ПОСЛЕ (image-based region fill) ===")
    print(after["overall_macro"])
    for c, v in after["per_category"].items():
        print(f"  {c:10s} f1={v['f1']:.3f}")


if __name__ == "__main__":
    main()
