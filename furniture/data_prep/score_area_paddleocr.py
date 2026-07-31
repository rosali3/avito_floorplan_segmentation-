"""
Тот же тест, что score_area_ocr.py, но с PaddleOCR вместо EasyOCR — сравниваем
детекторы текста на одной и той же ручной разметке (furniture/raw/
manual_area_gt_template.csv).

Запуск:
    python furniture/data_prep/score_area_paddleocr.py \
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
from paddleocr import PaddleOCR

HEIGHT_RE = re.compile(r"^h\s*=", re.IGNORECASE)
DECIMAL_RE = re.compile(r"^\d{1,2}[.,]\d{1,2}$")


def enhance_contrast(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    return cv2.addWeighted(out, 1.5, blur, -0.5, 0)


def ocr_all_numbers(ocr, image_bgr, enhance=False) -> list[float]:
    if enhance:
        image_bgr = enhance_contrast(image_bgr)
    result = ocr.predict(image_bgr)
    vals = []
    for page in result:
        texts = page.get("rec_texts", [])
        for text in texts:
            t = text.strip().replace(",", ".")
            if HEIGHT_RE.match(t) or not DECIMAL_RE.match(t):
                continue
            vals.append(float(t))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--enhance", action="store_true")
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

    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                    lang="ru", enable_mkldnn=False)
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
        detected = ocr_all_numbers(ocr, img, enhance=args.enhance)

        for row in rows:
            total += 1
            per_cat_total[row["category"]] += 1
            hit = any(abs(row["true_area_m2"] - d) <= 0.15 for d in detected)
            if hit:
                found += 1
                per_cat_found[row["category"]] += 1

        print(f"[score] {file_name}: true={sorted(r['true_area_m2'] for r in rows)} "
              f"detected={sorted(detected)}")

    print(f"\n=== PaddleOCR recall ===")
    print(f"Всего размечено вручную: {total}, найдено: {found} ({found/max(1,total):.1%})")
    print(f"\nПо категориям:")
    for cat in sorted(per_cat_total):
        t, fnd = per_cat_total[cat], per_cat_found[cat]
        print(f"  {cat:12s} {fnd}/{t} ({fnd/t:.1%})")


if __name__ == "__main__":
    main()
