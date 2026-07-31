"""
Быстрый эксперимент: закрашивает угол с водяным знаком Avito (логотип/текст
"Фото добавлено Авито", обычно в правом нижнем углу, иногда левом нижнем)
чёрным прямоугольником перед инференсом — проверяем, помогает ли это
YOLO/OCR или нет. Простая эвристика по фиксированной доле кадра, а не
детекция — специально для быстрого A/B теста.

Запуск:
    python data_prep/mask_watermark.py \
        --src data/ugc_test/images --dst data/ugc_test_nowm/images
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

WIDTH_FRAC = 0.30   # с обеих сторон (лого и текстовые варианты водяного знака)
HEIGHT_FRAC = 0.07


def mask_watermark(img):
    h, w = img.shape[:2]
    bh = int(h * HEIGHT_FRAC)
    bw = int(w * WIDTH_FRAC)
    img[h - bh:h, w - bw:w] = 0  # правый нижний угол (Avito лого)
    img[h - bh:h, 0:bw] = 0      # левый нижний угол ("Фото добавлено Авито")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    n = 0
    for f in sorted(src.iterdir()):
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = mask_watermark(img)
        # ВАЖНО: имя файла (включая расширение) должно совпадать 1-в-1 с
        # test_coco.json, иначе infer_and_eval не найдёт картинку
        ok = cv2.imwrite(str(dst / f.name), img)
        if not ok:
            print(f"[mask_watermark] не смог записать {f.name}, пропускаю")
            continue
        n += 1
    print(f"[mask_watermark] обработано {n} картинок -> {dst}")


if __name__ == "__main__":
    main()
