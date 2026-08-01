"""
Идея: комната на плане — это замкнутая область МЕЖДУ стенами (и проёмами —
дверь тоже часть границы, просто с разрывом). Вместо того чтобы доверять
рваным/дырявым blob'ам предсказанного класса комнаты напрямую, сначала
находим замкнутые области, используя предсказанные wall+opening как границу,
а затем заливаем каждую такую область ОДНИМ классом — тем, что встречается
внутри неё чаще всего среди предсказаний.

Работает на плотной карте меток (1 класс на пиксель, как в
compute_confusion_matrix.py: highest-score побеждает при перекрытии), а не
на списке инстансов — потому что "замкнутая область" по определению
пиксельное понятие.

Ограничение: если предсказанная стена сама дырявая (частый случай у
семантических моделей), соседние комнаты сольются в одну connected-component
через дыру — тогда голосование смешает классы двух реальных комнат. Не
панацея, а эксперимент.

Запуск:
    python eval/wall_bounded_fill.py --model segformer --score-thresh 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402
from compute_confusion_matrix import (  # noqa: E402
    gt_label_map, pred_label_map, split_ignore_regions, true_ignore_mask,
    ROOM_ID, LIVING_ID, BEDROOM_ID,
)
from mask_nms import mask_nms  # noqa: E402

WALL_ID, OPENING_ID = 6, 7
ROOM_TYPE_IDS = [1, 2, 3, 4, 5]  # living, bedroom, bathroom, kitchen, balcony


def wall_bounded_fill(label_map: np.ndarray, min_room_frac: float = 0.3) -> np.ndarray:
    """label_map: HxW, значения 0=background,1-7=классы (как из pred_label_map).
    Возвращает новую карту той же формы, комнатные connected-components
    залиты доминирующим room-type классом внутри них."""
    boundary = np.isin(label_map, [WALL_ID, OPENING_ID])
    interior = ~boundary
    labeled, n = ndimage.label(interior)
    out = label_map.copy()

    for comp_id in range(1, n + 1):
        comp_mask = labeled == comp_id
        vals = label_map[comp_mask]
        room_vals = vals[np.isin(vals, ROOM_TYPE_IDS)]
        if len(room_vals) == 0:
            continue
        counts = np.bincount(room_vals, minlength=8)
        majority = int(counts.argmax())
        if counts[majority] / len(vals) >= min_room_frac:
            out[comp_mask] = majority
    return out


def evaluate_dense(gt_map: np.ndarray, pred_map: np.ndarray, valid: np.ndarray,
                    agg: dict[int, dict[str, int]], class_ids: list[int]) -> None:
    for cid in class_ids:
        actual_gt_id = cid
        g = (gt_map == actual_gt_id) & valid
        if cid == ROOM_ID:
            p = np.isin(pred_map, [LIVING_ID, BEDROOM_ID]) & valid
        else:
            p = (pred_map == cid) & valid
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        agg[cid]["tp"] += tp
        agg[cid]["fp"] += fp
        agg[cid]["fn"] += fn


def summarize(agg: dict[int, dict[str, int]], names: dict[int, str]) -> dict:
    per_cat = {}
    for cid, a in agg.items():
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        denom_p, denom_r = tp + fp, tp + fn
        p = tp / denom_p if denom_p else float("nan")
        r = tp / denom_r if denom_r else float("nan")
        f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        per_cat[names[cid]] = {"iou": iou, "precision": p, "recall": r, "f1": f1}

    def macro(key):
        vals = [v[key] for v in per_cat.values() if v[key] == v[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    return {"per_category": per_cat,
            "overall_macro": {k: macro(k) for k in ("iou", "precision", "recall", "f1")}}


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

    agg_before = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids}
    agg_after = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids}

    for img in coco.loadImgs(coco.getImgIds()):
        img_id, h, w = img["id"], img["height"], img["width"]
        gt_map = gt_label_map(coco, img_id, h, w, room_by_img.get(img_id, []))
        pred_map = pred_label_map(preds, img_id, h, w, args.score_thresh)
        valid = ~true_ignore_mask(ignore_by_img, img_id, h, w)

        evaluate_dense(gt_map, pred_map, valid, agg_before, class_ids)
        filled = wall_bounded_fill(pred_map, min_room_frac=args.min_room_frac)
        evaluate_dense(gt_map, filled, valid, agg_after, class_ids)

    before = summarize(agg_before, names)
    after = summarize(agg_after, names)
    print("=== ДО (сырая dense-карта) ===")
    print(before["overall_macro"])
    for c, v in before["per_category"].items():
        print(f"  {c:10s} f1={v['f1']:.3f}")
    print("=== ПОСЛЕ (wall-bounded fill) ===")
    print(after["overall_macro"])
    for c, v in after["per_category"].items():
        print(f"  {c:10s} f1={v['f1']:.3f}")


if __name__ == "__main__":
    main()
