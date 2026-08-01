"""
Инференс обученной RF-DETR-Seg (Medium) на test-сплите (data/ugc_test/) +
подсчёт метрик через общий eval/coco_eval_common.py.

ВАЖНО про API: rfdetr .predict(image, threshold=...) в текущих версиях
возвращает объект в формате `supervision.Detections` с полями
.xyxy (N,4), .confidence (N,), .class_id (N,) и .mask (N,H,W bool) для
seg-моделей. Если в твоей версии библиотеки поля называются иначе — это
единственное место (функция `run_inference_on_image`), которое нужно поправить,
всё остальное (COCO-конвертация, eval) от точной сигнатуры не зависит.

class_id у supervision.Detections — 0-based индекс по порядку категорий в
_annotations.coco.json, на котором обучались (train/valid), поэтому мы
восстанавливаем исходный canonical category_id по этому же списку категорий,
а не считаем его равным class_id+1 "на глаз".

Запуск:
    python models/rfdetr_seg/infer_and_eval.py --checkpoint output/rfdetr_seg/checkpoints/best.pth
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from coco_utils import load_paths  # noqa: E402
from coco_eval_common import binary_mask_to_rle, run_coco_eval  # noqa: E402


def load_class_id_map(paths: dict) -> dict[int, int]:
    """0-based class_id (порядок в rfdetr train-датасете) -> canonical category_id."""
    train_ann = Path(paths["derived"]["rfdetr_dataset_dir"]) / "train" / "_annotations.coco.json"
    with open(train_ann, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    return {i: c["id"] for i, c in enumerate(cats)}


def run_inference_on_image(model, image: Image.Image, threshold: float):
    """Возвращает detections в унифицированном виде:
    list[dict(class_id_0based, score, mask HxW bool, bbox xyxy)]
    """
    detections = model.predict(image, threshold=threshold)
    out = []
    n = len(detections.xyxy)
    has_mask = getattr(detections, "mask", None) is not None
    for i in range(n):
        out.append({
            "class_id_0based": int(detections.class_id[i]),
            "score": float(detections.confidence[i]),
            "bbox_xyxy": detections.xyxy[i].tolist(),
            "mask": detections.mask[i] if has_mask else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--model-key", default="rfdetr_seg",
                     help="папка в output/ для предсказаний+метрик — задай отдельное имя "
                          "(напр. rfdetr_seg_fullaug), чтобы не перезаписать результаты другого чекпоинта")
    args = ap.parse_args()

    from rfdetr import RFDETRSegMedium

    paths = load_paths()
    class_id_map = load_class_id_map(paths)

    ugc_dir = Path(paths["derived"]["ugc_test_dir"])
    with open(ugc_dir / "test_coco.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    model = RFDETRSegMedium(pretrain_weights=args.checkpoint)

    predictions = []
    for img_rec in gt["images"]:
        img_path = ugc_dir / "images" / img_rec["file_name"]
        image = Image.open(img_path).convert("RGB")
        dets = run_inference_on_image(model, image, args.threshold)
        for d in dets:
            x0, y0, x1, y1 = d["bbox_xyxy"]
            bbox = [x0, y0, x1 - x0, y1 - y0]
            pred = {
                "image_id": img_rec["id"],
                "category_id": class_id_map[d["class_id_0based"]],
                "bbox": bbox,
                "score": d["score"],
            }
            if d["mask"] is not None:
                pred["segmentation"] = binary_mask_to_rle(np.asarray(d["mask"]))
            else:
                # если модель почему-то не отдала маску, используем bbox как
                # прямоугольную маску — грубое приближение, лучше чем падение
                mask = np.zeros((img_rec["height"], img_rec["width"]), dtype=np.uint8)
                xi0, yi0, xi1, yi1 = map(int, [x0, y0, x1, y1])
                mask[max(0, yi0):yi1, max(0, xi0):xi1] = 1
                pred["segmentation"] = binary_mask_to_rle(mask)
            predictions.append(pred)

    out_dir = Path(paths["derived"]["output_dir"]) / args.model_key / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[rfdetr infer] {len(predictions)} predictions -> {pred_path}")

    gt_path = ugc_dir / "test_coco.json"
    for iou_type in ("segm", "bbox"):
        metrics = run_coco_eval(gt_path, predictions, iou_type=iou_type)
        with open(out_dir / f"metrics_{iou_type}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[rfdetr infer] {iou_type}: AP@[.5:.95]={metrics['overall']['AP@[.5:.95]']:.4f} "
              f"AP@.50={metrics['overall']['AP@.50']:.4f}")


if __name__ == "__main__":
    main()
