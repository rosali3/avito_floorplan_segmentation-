"""
Надстройка над готовой моделью сегментации комнат: для каждого предсказанного
(или GT) инстанса комнаты находит на плане подпись площади через OCR
(EasyOCR, кириллица) + пространственную привязку "число внутри полигона
комнаты".

Почему не просто регексом по всему тексту картинки: подписи площади и высоты
потолка визуально похожи (оба — decimal-числа), а сама OCR-модель на реальных
UGC-фото ненадёжно разделяет строки дроби "номер комнаты / площадь" — иногда
сливает их в один искажённый токен, иногда номер комнаты вообще не
детектируется. См. обсуждение в furniture/data_prep/ocr_visualize.py.
Поэтому единственный надёжный сигнал — гео­метрия: decimal-число БЕЗ префикса
h=/H=, чей центр лежит внутри полигона комнаты, ближайшее к её центроиду.

Вход — COCO-формат предсказаний (тот же, что во всех models/*/infer_and_eval.py)
или сам test_coco.json (GT) — оба используют одинаковый формат instance-масок,
поэтому скрипт единый для проверки на GT и на реальных предсказаниях.

Запуск:
    python furniture/data_prep/room_area_ocr.py \
        --gt data/ugc_test/test_coco.json \
        --images-dir data/ugc_test/images \
        --n 6 --out-dir furniture/raw/room_area_vis
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
from pycocotools import mask as mask_utils

HEIGHT_RE = re.compile(r"^h\s*=", re.IGNORECASE)
DECIMAL_RE = re.compile(r"^\d{1,2}[.,]\d{1,2}$")
MIN_AREA_M2 = 0.5
MAX_AREA_M2 = 60.0


def polygon_to_mask(seg, h, w) -> np.ndarray:
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def area_candidates_from_ocr(results) -> list[dict]:
    """Все decimal-числа без h=/H=, в правдоподобном диапазоне площади."""
    cands = []
    for bbox, text, conf in results:
        t = text.strip().replace(",", ".")
        if HEIGHT_RE.match(t) or not DECIMAL_RE.match(t):
            continue
        val = float(t)
        if not (MIN_AREA_M2 <= val <= MAX_AREA_M2):
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cands.append({"value": val, "cx": (min(xs) + max(xs)) / 2,
                      "cy": (min(ys) + max(ys)) / 2, "conf": conf, "text": text})
    return cands


def ocr_crop_per_room(reader, img: np.ndarray, instances: list[dict], h: int, w: int,
                       pad_frac: float = 0.15, crop_upscale: float = 3.0) -> list[dict]:
    """Вместо OCR по всей картинке (мелкий шрифт площади теряется на общем фоне)
    кропаем bbox каждой комнаты (+padding) и апскейлим ИМЕННО кроп — маленькая
    картинка, можно смело растягивать в 3-4 раза без риска OOM, а текст внутри
    становится в разы крупнее относительно кадра. Кандидаты всё равно
    проверяются на попадание в маску комнаты (не просто в bbox), чтобы padding
    не утащил число соседней комнаты."""
    out = []
    for inst in instances:
        mask = polygon_to_mask(inst["segmentation"], h, w)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            out.append({**inst, "area_m2": None})
            continue
        cx0, cy0 = xs.mean(), ys.mean()

        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad_x, pad_y = int((x1 - x0) * pad_frac), int((y1 - y0) * pad_frac)
        cx0i, cx1i = max(0, x0 - pad_x), min(w, x1 + pad_x + 1)
        cy0i, cy1i = max(0, y0 - pad_y), min(h, y1 + pad_y + 1)
        crop = img[cy0i:cy1i, cx0i:cx1i]
        if crop.size == 0:
            out.append({**inst, "area_m2": None})
            continue

        big = cv2.resize(crop, None, fx=crop_upscale, fy=crop_upscale, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(big)
        results_full = [
            ([[p[0] / crop_upscale + cx0i, p[1] / crop_upscale + cy0i] for p in bbox], text, conf)
            for bbox, text, conf in results
        ]
        candidates = area_candidates_from_ocr(results_full)
        inside = [c for c in candidates if mask[int(round(min(max(c["cy"], 0), h - 1))),
                                                 int(round(min(max(c["cx"], 0), w - 1)))]]
        if not inside:
            out.append({**inst, "area_m2": None})
            continue
        best = min(inside, key=lambda c: (c["cx"] - cx0) ** 2 + (c["cy"] - cy0) ** 2)
        out.append({**inst, "area_m2": best["value"], "area_ocr_text": best["text"], "area_ocr_conf": best["conf"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="COCO json с instance-масками (GT или предсказания модели)")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out-dir", default="furniture/raw/room_area_vis")
    ap.add_argument("--crop-upscale", type=float, default=3.0,
                     help="апскейл КРОПА каждой комнаты перед OCR (не всей картинки — "
                          "иначе мелкий шрифт площади теряется на общем фоне и легко ловим OOM)")
    args = ap.parse_args()

    with open(args.gt, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    # площадь печатают только внутри жилых помещений — у wall/opening подписи
    # площади в принципе не бывает, гонять по ним OCR бессмысленно
    room_cat_ids = {cid for cid, name in cat_names.items() if name not in ("wall", "opening")}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in room_cat_ids:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    reader = easyocr.Reader(["ru", "en"], gpu=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.images_dir)

    for img_rec in coco["images"][: args.n]:
        instances = anns_by_img.get(img_rec["id"], [])
        if not instances:
            continue
        img_path = img_dir / img_rec["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img_rec["height"], img_rec["width"]

        rooms = ocr_crop_per_room(reader, img, instances, h, w, crop_upscale=args.crop_upscale)

        vis = img.copy()
        n_found = 0
        for room in rooms:
            mask = polygon_to_mask(room["segmentation"], h, w)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            color = (0, 165, 255) if room["area_m2"] is None else (0, 200, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
            name = cat_names.get(room["category_id"], "?")
            label = f"{name}: {room['area_m2']} m2" if room["area_m2"] is not None else f"{name}: ?"
            if room["area_m2"] is not None:
                n_found += 1
            cv2.putText(vis, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        out_path = out_dir / f"{Path(img_rec['file_name']).stem}_area.png"
        cv2.imwrite(str(out_path), vis)
        print(f"[room_area_ocr] {img_rec['file_name']}: {n_found}/{len(rooms)} комнат с площадью -> {out_path}")


if __name__ == "__main__":
    main()
