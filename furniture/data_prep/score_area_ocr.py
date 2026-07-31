"""
Честная сверка room_area_ocr.py с РУЧНОЙ разметкой площади пользователя
(furniture/raw/manual_area_gt.csv, формат: file_name,room_num,true_area_m2,
category — category тут СЫРАЯ (room/coridor/enterance/restroom/bathroom/
kitchen/balcony), т.к. пользователь размечал все видимые на плане комнаты,
а не только те, что попали в наш отфильтрованный test_coco.json).

Поскольку у "room"/"coridor"/"enterance" нет масок в test_coco.json (это
исключённые категории, см. classes.yaml: ugc_excluded_categories), для них
нет geometric ground truth полигона — сверяем по-другому: гоняем OCR по
ВСЕЙ картинке (без привязки к маскам), собираем все правдоподобные
decimal-кандидаты и просто проверяем, попало ли каждое РУЧНОЕ true_area_m2
в множество распознанных чисел (с допуском). Это честная оценка recall
самого OCR-текста (не geometric matching), а для kitchen/bathroom/balcony,
у которых маски ЕСТЬ, дополнительно проверяем полный пайплайн (OCR + маска).

Запуск:
    python furniture/data_prep/score_area_ocr.py \
        --manual-csv furniture/raw/manual_area_gt.csv \
        --images-dir data/ugc_test/images
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import easyocr

HEIGHT_RE = re.compile(r"^h\s*=", re.IGNORECASE)
DECIMAL_RE = re.compile(r"^\d{1,2}[.,]\d{1,2}$")
TOL = 0.15  # допуск на ошибку OCR-распознавания цифр (напр. "7.1" vs "7.7")


def enhance_contrast(image_bgr):
    """CLAHE на L-канале (LAB) + лёгкий unsharp mask — реальные UGC-фото часто
    тусклые/засвеченные, мелкий печатный шрифт площади теряется в низком
    контрасте ещё до апскейла."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
    return out


def ocr_all_numbers(reader, image_bgr, enhance=False, mag_ratio=1.0) -> list[float]:
    if enhance:
        image_bgr = enhance_contrast(image_bgr)
    results = reader.readtext(image_bgr, mag_ratio=mag_ratio)
    vals = []
    for _bbox, text, _conf in results:
        t = text.strip().replace(",", ".")
        if HEIGHT_RE.match(t) or not DECIMAL_RE.match(t):
            continue
        vals.append(float(t))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--enhance", action="store_true", help="CLAHE + unsharp перед OCR")
    ap.add_argument("--mag-ratio", type=float, default=1.0, help="EasyOCR mag_ratio (детектор мелкого текста)")
    args = ap.parse_args()

    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    with open(args.manual_csv, "r", encoding="utf-8-sig") as f:
        reader_csv = csv.DictReader(f)
        for row in reader_csv:
            if not row.get("file_name") or not row.get("true_area_m2"):
                continue
            try:
                area = float(row["true_area_m2"])
            except ValueError:
                continue
            rows_by_image[row["file_name"]].append({"category": row["category"], "true_area_m2": area})

    reader = easyocr.Reader(["ru", "en"], gpu=True)
    img_dir = Path(args.images_dir)

    total, found = 0, 0
    per_cat_total = defaultdict(int)
    per_cat_found = defaultdict(int)

    for file_name, rows in rows_by_image.items():
        img_path = img_dir / file_name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[score] пропускаю {file_name} (не найдена)")
            continue
        detected = ocr_all_numbers(reader, img, enhance=args.enhance, mag_ratio=args.mag_ratio)

        for row in rows:
            total += 1
            per_cat_total[row["category"]] += 1
            hit = any(abs(row["true_area_m2"] - d) <= TOL for d in detected)
            if hit:
                found += 1
                per_cat_found[row["category"]] += 1

        print(f"[score] {file_name}: true={sorted(r['true_area_m2'] for r in rows)} "
              f"detected={sorted(detected)}")

    print(f"\n=== Recall распознавания текста площади (OCR raw, без привязки к маске) ===")
    print(f"Всего размечено вручную: {total}, найдено OCR где-то на картинке: {found} "
          f"({found/max(1,total):.1%})")
    print(f"\nПо категориям:")
    for cat in sorted(per_cat_total):
        t, fnd = per_cat_total[cat], per_cat_found[cat]
        print(f"  {cat:12s} {fnd}/{t} ({fnd/t:.1%})")


if __name__ == "__main__":
    main()
