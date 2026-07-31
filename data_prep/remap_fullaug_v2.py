"""
Ремаппинг combined_out_v2 (9-классовая таксономия коллеги: door=7 и
window_ext=8 разделены) в 7-классовую таксономию проекта (opening=7, где
opening = двери + окна), чтобы combined_out_v2 можно было скормить тем же
data_prep/build_train_val_coco.py / classes.yaml, что и обычный combined_out.

Картинки (images/) НЕ меняются — только их пиксельный состав в масках не
трогаем, поэтому просто симлинкаем (экономит место и время копирования).
Маски (masks/) переписываются: пиксель==8 (window_ext) -> 7 (opening),
всё остальное как есть.

Запуск (на сервере, после того как combined_out_v2 залит и распакован):
    python data_prep/remap_fullaug_v2.py \
        --src /path/to/combined_out_v2 \
        --dst /path/to/combined_out_v2_remapped
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

WINDOW_EXT_ID = 8
OPENING_ID = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    src_images = src / "images"
    src_masks = src / "masks"
    sources = sorted(p.name for p in src_images.iterdir() if p.is_dir())
    print(f"[remap_fullaug_v2] источники: {sources}")

    n_total, n_changed = 0, 0
    for source_name in sources:
        img_dir = src_images / source_name
        mask_dir = src_masks / source_name
        dst_img_dir = dst / "images" / source_name
        dst_mask_dir = dst / "masks" / source_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_mask_dir.mkdir(parents=True, exist_ok=True)

        mask_files = sorted(mask_dir.glob("*.png"))
        for mask_path in tqdm(mask_files, desc=f"remap {source_name}"):
            n_total += 1
            name = mask_path.name
            img_path = img_dir / name
            if not img_path.is_file():
                continue

            dst_img_path = dst_img_dir / name
            if not dst_img_path.exists():
                os.symlink(img_path.resolve(), dst_img_path)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask.ndim != 2:
                mask = mask[:, :, 0]
            if (mask == WINDOW_EXT_ID).any():
                mask = mask.copy()
                mask[mask == WINDOW_EXT_ID] = OPENING_ID
                n_changed += 1
            cv2.imwrite(str(dst_mask_dir / name), mask)

    print(f"[remap_fullaug_v2] done. всего масок={n_total}, "
          f"с window_ext(->opening)={n_changed} -> {dst}")


if __name__ == "__main__":
    main()
