"""
4 варианта детектирования "замкнутых пространств" (комнат между стенами) —
сравнение на UNet-simple, самой пострадавшей от исходного wall_bounded_fill.py.

1. connected_components — cv2.connectedComponents, функциональный аналог
   scipy.ndimage.label() из wall_bounded_fill.py (базовый вариант, для сверки).
2. contours — cv2.findContours(RETR_CCOMP): комнаты = "дырки" внутри
   контура стены (иерархия контуров). Та же уязвимость к разрывам, просто
   другой код.
3. morph_closing — cv2.morphologyEx(MORPH_CLOSE) заращивает мелкие разрывы
   границы ПЕРЕД разметкой связных компонент.
4. watershed — cv2.distanceTransform + cv2.watershed: затравки в самых
   "глубоких" точках (дальше всего от стены), маркерный watershed не сливает
   две разные затравки, даже если между ними формально есть путь через
   разрыв — принципиально устойчивее к дырам.

Запуск:
    python eval/room_fill_variants.py --model unet_baseline
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
from wall_bounded_fill import evaluate_dense, summarize  # noqa: E402
from mask_nms import mask_nms  # noqa: E402

WALL_ID, OPENING_ID = 6, 7
ROOM_TYPE_IDS = [1, 2, 3, 4, 5]


def _majority_fill(label_map: np.ndarray, comp_mask: np.ndarray, min_room_frac: float) -> int | None:
    vals = label_map[comp_mask]
    room_vals = vals[np.isin(vals, ROOM_TYPE_IDS)]
    if len(room_vals) == 0:
        return None
    counts = np.bincount(room_vals, minlength=8)
    majority = int(counts.argmax())
    return majority if counts[majority] / len(vals) >= min_room_frac else None


def fill_connected_components(label_map: np.ndarray, min_room_frac: float = 0.3) -> np.ndarray:
    boundary = np.isin(label_map, [WALL_ID, OPENING_ID])
    interior = (~boundary).astype(np.uint8)
    n, labels = cv2.connectedComponents(interior, connectivity=8)
    out = label_map.copy()
    for comp_id in range(1, n):
        m = labels == comp_id
        cls = _majority_fill(label_map, m, min_room_frac)
        if cls is not None:
            out[m] = cls
    return out


def fill_contours(label_map: np.ndarray, min_room_frac: float = 0.3) -> np.ndarray:
    boundary = (np.isin(label_map, [WALL_ID, OPENING_ID]).astype(np.uint8)) * 255
    contours, hierarchy = cv2.findContours(boundary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = label_map.copy()
    if hierarchy is None:
        return out
    hierarchy = hierarchy[0]
    for i, h in enumerate(hierarchy):
        if h[3] == -1:  # внешний контур самой стены, не "дырка"/комната
            continue
        mask = np.zeros(label_map.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, i, 1, thickness=-1)
        m = mask.astype(bool)
        cls = _majority_fill(label_map, m, min_room_frac)
        if cls is not None:
            out[m] = cls
    return out


def fill_morph_closing(label_map: np.ndarray, min_room_frac: float = 0.3, kernel_size: int = 5) -> np.ndarray:
    boundary = (np.isin(label_map, [WALL_ID, OPENING_ID]).astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, kernel)
    interior = (closed == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(interior, connectivity=8)
    out = label_map.copy()
    for comp_id in range(1, n):
        m = labels == comp_id
        cls = _majority_fill(label_map, m, min_room_frac)
        if cls is not None:
            out[m] = cls
    return out


def fill_watershed(label_map: np.ndarray, min_room_frac: float = 0.3, dist_frac: float = 0.3) -> np.ndarray:
    boundary = np.isin(label_map, [WALL_ID, OPENING_ID])
    interior = (~boundary).astype(np.uint8)
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    out = label_map.copy()
    if dist.max() == 0:
        return out

    sure_fg = (dist > dist_frac * dist.max()).astype(np.uint8)
    n_markers, seed_labels = cv2.connectedComponents(sure_fg, connectivity=8)
    if n_markers <= 1:
        return out

    markers = (seed_labels + 1).astype(np.int32)  # затравки: 2,3,4,... (1 зарезервирован под фон/стену)
    unknown = (interior == 1) & (sure_fg == 0)
    markers[unknown] = 0
    markers[boundary] = 1

    img3 = cv2.cvtColor((interior * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)

    for room_label in range(2, n_markers + 1):
        m = markers == room_label
        if not m.any():
            continue
        cls = _majority_fill(label_map, m, min_room_frac)
        if cls is not None:
            out[m] = cls
    return out


VARIANTS = {
    "connected_components": fill_connected_components,
    "contours": fill_contours,
    "morph_closing": fill_morph_closing,
    "watershed": fill_watershed,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-room-frac", type=float, default=0.3)
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

    variant_names = ["baseline (без заливки)"] + list(VARIANTS.keys())
    aggs = {v: {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids} for v in variant_names}

    for img in coco.loadImgs(coco.getImgIds()):
        img_id, h, w = img["id"], img["height"], img["width"]
        gtm = gt_label_map(coco, img_id, h, w, room_by_img.get(img_id, []))
        pm = pred_label_map(preds, img_id, h, w, args.score_thresh)
        valid = ~true_ignore_mask(ignore_by_img, img_id, h, w)

        evaluate_dense(gtm, pm, valid, aggs["baseline (без заливки)"], class_ids)
        for name, fn in VARIANTS.items():
            filled = fn(pm, min_room_frac=args.min_room_frac)
            evaluate_dense(gtm, filled, valid, aggs[name], class_ids)

    print(f"{'Вариант':28s} {'IoU':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for name in variant_names:
        s = summarize(aggs[name], names)
        o = s["overall_macro"]
        print(f"{name:28s} {o['iou']:6.3f} {o['precision']:6.3f} {o['recall']:6.3f} {o['f1']:6.3f}")
        for c, v in s["per_category"].items():
            if v["f1"] == v["f1"]:
                print(f"    {c:10s} f1={v['f1']:.3f}")


if __name__ == "__main__":
    main()
