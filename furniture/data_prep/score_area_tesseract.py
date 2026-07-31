"""
Тот же тест, что score_area_ocr.py/score_area_paddleocr.py, но Tesseract с
whitelist только цифр+точка/запятая (--psm 7, одна строка) — идея: раз нам
нужны только числа площади (не кириллица), сузить словарь и посмотреть,
даёт ли это прирост за счёт устранения путаницы с буквами.

Гоняем по тем же кропам комнат (crop+upscale), что лучше всего работало у
EasyOCR — без привязки к маске, просто по bbox всей картинки, т.к. Tesseract
тут используется как recognition-only (без своего text-детектора, psm=7
ожидает уже вырезанную строку) — сравниваем на скользящем окне вокруг
изображения было бы избыточно, поэтому просто прогоняем ВСЮ картинку с
psm=11 (sparse text) как более честный аналог "детектор+распознавание".

Запуск:
    python furniture/data_prep/score_area_tesseract.py \
        --manual-csv furniture/raw/manual_area_gt_template.csv \
        --images-dir data/ugc_test/images
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DECIMAL_RE = re.compile(r"^\d{1,2}[.,]\d{1,2}$")


def ocr_all_numbers(image_bgr) -> list[float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    config = r'--psm 11 -c tessedit_char_whitelist=0123456789.,'
    text = pytesseract.image_to_string(big, config=config)
    vals = []
    for tok in re.split(r"\s+", text.strip()):
        t = tok.strip().replace(",", ".")
        if DECIMAL_RE.match(t):
            vals.append(float(t))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-csv", required=True)
    ap.add_argument("--images-dir", required=True)
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

    img_dir = Path(args.images_dir)
    total, found = 0, 0
    per_cat_total = defaultdict(int)
    per_cat_found = defaultdict(int)

    for file_name, rows in rows_by_image.items():
        img_path = img_dir / file_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        detected = ocr_all_numbers(img)

        for row in rows:
            total += 1
            per_cat_total[row["category"]] += 1
            hit = any(abs(row["true_area_m2"] - d) <= 0.15 for d in detected)
            if hit:
                found += 1
                per_cat_found[row["category"]] += 1

        print(f"[score] {file_name}: true={sorted(r['true_area_m2'] for r in rows)} detected={sorted(detected)}")

    print(f"\n=== Tesseract (digit-only, psm=11) recall ===")
    print(f"Всего размечено вручную: {total}, найдено: {found} ({found/max(1,total):.1%})")
    print(f"\nПо категориям:")
    for cat in sorted(per_cat_total):
        t, fnd = per_cat_total[cat], per_cat_found[cat]
        print(f"  {cat:12s} {fnd}/{t} ({fnd/t:.1%})")


if __name__ == "__main__":
    main()
