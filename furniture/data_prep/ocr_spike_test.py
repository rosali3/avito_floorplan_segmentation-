"""
Быстрый спайк: проверить, читает ли готовый OCR (EasyOCR, кириллица) подписи
площади ("12.5 м²") и названий комнат прямо на реальных UGC-планах. Ничего не
обучаем — только инференс готовой моделью на нескольких тестовых картинках,
чтобы решить, стоит ли закладывать OCR в финальный пайплайн.

Запуск:
    python furniture/data_prep/ocr_spike_test.py --n 8
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import easyocr

AREA_RE = re.compile(r"\d+[.,]\d+\s*м?2?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/ugc_test/images")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    reader = easyocr.Reader(["ru", "en"], gpu=True)

    img_dir = Path(args.images_dir)
    files = sorted(img_dir.iterdir())[: args.n]

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[skip] cannot read {f.name}")
            continue
        results = reader.readtext(img)
        print(f"\n=== {f.name} ({img.shape[1]}x{img.shape[0]}) ===")
        if not results:
            print("  (ничего не распознано)")
            continue
        for bbox, text, conf in results:
            tag = " <-- похоже на площадь" if AREA_RE.search(text) else ""
            print(f"  [{conf:.2f}] {text!r}{tag}")


if __name__ == "__main__":
    main()
