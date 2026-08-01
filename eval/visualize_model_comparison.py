"""
Собирает PNG-коллажи: для каждой UGC test-картинки — панели с семантическими
масками GT и предсказаний каждой доступной модели, наложенными на фото.

Запуск (одна картинка):
    python eval/visualize_model_comparison.py --image-substr _pR8Mra4Un1Km5
Запуск (все картинки test-сплита):
    python eval/visualize_model_comparison.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from coco_utils import load_paths  # noqa: E402

MODEL_DIRS = [
    ("rfdetr_seg", "RF-DETR-Seg"),
    ("rfdetr_seg_fullaug", "RF-DETR fullaug"),
    ("yolo_seg", "YOLO-seg"),
    ("yolo_seg_fullaug", "YOLO fullaug"),
    ("maskrcnn_mmdet", "Mask R-CNN"),
    ("segformer", "SegFormer"),
    ("sam_zeroshot", "SAM zero-shot"),
    ("sam_finetuned", "SAM fine-tuned"),
    ("unet_baseline", "UNet-simple"),
]

# id класса -> цвет (RGB 0..255)
PALETTE = {
    1: (255, 99, 71),    # living
    2: (60, 179, 113),   # bedroom
    3: (65, 105, 225),   # bathroom
    4: (255, 215, 0),    # kitchen
    5: (238, 130, 238),  # balcony
    6: (128, 128, 128),  # wall
    7: (255, 140, 0),    # opening
}

CLASS_NAMES = {1: "living", 2: "bedroom", 3: "bathroom", 4: "kitchen",
               5: "balcony", 6: "wall", 7: "opening"}


def legend_handles():
    return [mpatches.Patch(color=tuple(c / 255 for c in PALETTE[cid]), label=name)
            for cid, name in CLASS_NAMES.items()]


def ann_or_pred_to_mask(seg, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def semantic_overlay(image_bgr: np.ndarray, class_masks: dict[int, np.ndarray], alpha: float = 0.5) -> np.ndarray:
    out = image_bgr.copy().astype(np.float32)
    for cid, m in class_masks.items():
        color = np.array(PALETTE[cid][::-1], dtype=np.float32)  # RGB->BGR
        out[m] = out[m] * (1 - alpha) + color * alpha
    return out.astype(np.uint8)


def draw_ignore_hatch(image_bgr: np.ndarray, ignore_mask: np.ndarray, spacing: int = 8) -> np.ndarray:
    """Штриховка (диагональные линии) поверх ignore-зон (room/coridor/hall/...),
    чтобы визуально отличать "не размечено" от реального фона."""
    if not ignore_mask.any():
        return image_bgr
    out = image_bgr.copy()
    h, w = ignore_mask.shape
    hatch = np.zeros((h, w), dtype=bool)
    for offset in range(-h, w, spacing):
        ys = np.arange(h)
        xs = ys + offset
        valid = (xs >= 0) & (xs < w)
        hatch[ys[valid], xs[valid]] = True
    m = ignore_mask & hatch
    out[m] = (out[m].astype(np.float32) * 0.3 + np.array([40, 40, 40], dtype=np.float32) * 0.7).astype(np.uint8)
    # лёгкая общая дымка на всей ignore-зоне, чтобы граница была видна даже без штриха
    dim = ignore_mask & ~hatch
    out[dim] = (out[dim].astype(np.float32) * 0.75).astype(np.uint8)
    return out


def gt_class_masks(coco: COCO, img_id: int, h: int, w: int) -> dict[int, np.ndarray]:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    out: dict[int, np.ndarray] = {}
    for ann in anns:
        m = ann_or_pred_to_mask(ann["segmentation"], h, w)
        out[ann["category_id"]] = out.get(ann["category_id"], np.zeros((h, w), bool)) | m
    return out


def pred_class_masks(preds: list[dict], img_id: int, h: int, w: int, score_thresh: float) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for p in preds:
        if p["image_id"] != img_id or p.get("score", 1.0) < score_thresh:
            continue
        m = ann_or_pred_to_mask(p["segmentation"], h, w)
        out[p["category_id"]] = out.get(p["category_id"], np.zeros((h, w), bool)) | m
    return out


def load_ignore_regions_by_img(gt_json_path: Path) -> dict[int, list[dict]]:
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    by_img: dict[int, list[dict]] = {}
    for region in gt.get("ignore_regions", []):
        by_img.setdefault(region["image_id"], []).append(region)
    return by_img


def ignore_mask_for_image(ignore_by_img: dict[int, list[dict]], img_id: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for region in ignore_by_img.get(img_id, []):
        m |= ann_or_pred_to_mask(region["segmentation"], h, w)
    return m


def load_predictions_cache(output_dir: Path) -> dict[str, list[dict] | None]:
    cache: dict[str, list[dict] | None] = {}
    for model_key, _ in MODEL_DIRS:
        pred_path = output_dir / model_key / "predictions" / "test_predictions.json"
        if pred_path.is_file():
            with open(pred_path, "r", encoding="utf-8") as f:
                cache[model_key] = json.load(f)
        else:
            cache[model_key] = None
    return cache


def build_collage(coco: COCO, img_info: dict, ugc_test_dir: Path,
                   score_thresh: float, out_path: Path, preds_cache: dict,
                   ignore_by_img: dict[int, list[dict]],
                   thresh_overrides: dict[str, float] | None = None,
                   gt_out_dir: Path | None = None) -> None:
    thresh_overrides = thresh_overrides or {}
    img_id, h, w = img_info["id"], img_info["height"], img_info["width"]
    image_bgr = cv2.imread(str(ugc_test_dir / "images" / img_info["file_name"]))
    if image_bgr is None:
        print(f"[visualize_model_comparison] пропуск (не читается): {img_info['file_name']}")
        return

    gt_overlay = semantic_overlay(image_bgr, gt_class_masks(coco, img_id, h, w))
    ig_mask = ignore_mask_for_image(ignore_by_img, img_id, h, w)
    gt_overlay = draw_ignore_hatch(gt_overlay, ig_mask)
    panels = [("GT (разметка, штрих = ignore: room/hall/...)", gt_overlay)]

    if gt_out_dir is not None:
        gt_out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(gt_out_dir / f"{Path(img_info['file_name']).stem}.png"), gt_overlay)

    for model_key, label in MODEL_DIRS:
        preds = preds_cache.get(model_key)
        if preds is None:
            panels.append((f"{label}\n(нет предсказаний)", image_bgr.copy()))
            continue
        thresh = thresh_overrides.get(model_key, score_thresh)
        cmasks = pred_class_masks(preds, img_id, h, w, thresh)
        title = label if model_key not in thresh_overrides else f"{label} (thr={thresh})"
        panels.append((title, semantic_overlay(image_bgr, cmasks)))

    n = len(panels)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    for ax, (label, panel) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(img_info["file_name"], fontsize=9)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=len(CLASS_NAMES),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[visualize_model_comparison] -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-substr", default=None, help="подстрока имени файла для одной картинки")
    ap.add_argument("--all", action="store_true", help="сгенерировать коллаж для каждой картинки test-сплита")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--out", default="docs/report_assets/model_comparison_visual.png",
                     help="путь для --image-substr; для --all используется как папка")
    ap.add_argument("--out-dir", default="docs/report_assets/ugc_comparisons",
                     help="папка для коллажей при --all")
    ap.add_argument("--gt-out-dir", default="docs/report_assets/ugc_gt_masks",
                     help="папка для отдельных GT-наложений (без сетки моделей)")
    ap.add_argument("--rfdetr-thresh", type=float, default=0.1,
                     help="отдельный (пониженный) порог confidence для RF-DETR")
    ap.add_argument("--yolo-thresh", type=float, default=0.1,
                     help="отдельный (пониженный) порог confidence для YOLO")
    args = ap.parse_args()

    thresh_overrides = {
        "rfdetr_seg": args.rfdetr_thresh, "rfdetr_seg_fullaug": args.rfdetr_thresh,
        "yolo_seg": args.yolo_thresh, "yolo_seg_fullaug": args.yolo_thresh,
    }

    if not args.all and not args.image_substr:
        raise SystemExit("укажи --image-substr <подстрока> или --all")

    paths = load_paths()
    ugc_test_dir = Path(paths["derived"]["ugc_test_dir"])
    output_dir = Path(paths["derived"]["output_dir"])

    coco = COCO(str(ugc_test_dir / "test_coco.json"))
    preds_cache = load_predictions_cache(output_dir)
    ignore_by_img = load_ignore_regions_by_img(ugc_test_dir / "test_coco.json")
    available = [k for k, v in preds_cache.items() if v is not None]
    missing = [k for k, v in preds_cache.items() if v is None]
    print(f"[visualize_model_comparison] предсказания есть: {available}")
    print(f"[visualize_model_comparison] предсказаний нет: {missing}")

    all_imgs = coco.loadImgs(coco.getImgIds())

    if args.all:
        out_dir = Path(args.out_dir)
        gt_out_dir = Path(args.gt_out_dir)
        for img_info in all_imgs:
            stem = Path(img_info["file_name"]).stem
            build_collage(coco, img_info, ugc_test_dir, args.score_thresh,
                          out_dir / f"{stem}.png", preds_cache, ignore_by_img,
                          thresh_overrides=thresh_overrides, gt_out_dir=gt_out_dir)
        print(f"[visualize_model_comparison] всего: {len(all_imgs)} -> {out_dir}")
        print(f"[visualize_model_comparison] GT-наложения отдельно -> {gt_out_dir}")
    else:
        img_info = next((im for im in all_imgs if args.image_substr in im["file_name"]), None)
        if img_info is None:
            raise SystemExit(f"картинка с подстрокой {args.image_substr!r} не найдена")
        build_collage(coco, img_info, ugc_test_dir, args.score_thresh, Path(args.out), preds_cache,
                      ignore_by_img, thresh_overrides=thresh_overrides, gt_out_dir=Path(args.gt_out_dir))


if __name__ == "__main__":
    main()
