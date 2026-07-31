"""
Собирает YOLO-seg датасет мебели из parsed_furniture.json (сейчас только
SFPI — CubiCasa5K временно исключён из-за нерешённого расхождения координат
SVG/PNG, см. furniture/data_prep/parse_cubicasa_svg.py и commit message).

TIFF -> JPG (значительно компактнее, ultralytics с TIFF не работает из коробки).
Использует собственный train/val сплит SFPI (test не берём — не нужен для
обучения детектора).

Запуск:
    python furniture/data_prep/build_furniture_yolo.py \
        --parsed furniture/raw/sfpi/parsed_furniture.json \
        --images-root furniture/raw/sfpi/images_extracted/SFPI/Images \
        --out furniture/data/furniture_yolo_ds
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

SPLIT_MAP = {"train": "train", "val": "val"}  # test не берём


def add_cubicasa(out_root: Path, cubi_json: Path, cubi_root: Path, name_to_idx: dict,
                  cubi_to_canon: dict, split_seed: int, train_frac: float):
    records = json.load(open(cubi_json, "r", encoding="utf-8"))
    rng = random.Random(split_seed)
    n_written = 0
    for rec in records:
        split = "train" if rng.random() < train_frac else "val"
        raster = cubi_root / rec["raster_path"]
        if not raster.is_file():
            continue
        img = cv2.imread(str(raster), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = rec["height"], rec["width"]

        lines = []
        for inst in rec["instances"]:
            canon = cubi_to_canon.get(inst["class_name"])
            if canon is None:
                continue
            cls_idx = name_to_idx.get(canon)
            if cls_idx is None:
                continue
            pts = np.array(inst["polygon"], dtype=np.float64).reshape(-1, 2)
            pts[:, 0] /= w
            pts[:, 1] /= h
            pts = np.clip(pts, 0.0, 1.0)
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{cls_idx} {coords}")
        if not lines:
            continue

        stem = "cubi_" + Path(rec["svg_path"]).parent.name
        img_out_dir = out_root / "images" / split
        lbl_out_dir = out_root / "labels" / split
        cv2.imwrite(str(img_out_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (lbl_out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_written += 1
    print(f"[build_furniture_yolo] CubiCasa5K: добавлено {n_written} изображений")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--images-root", required=True, help="папка с train/val/test.tiff внутри")
    ap.add_argument("--classes-yaml", default="furniture/configs/furniture_classes.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cubicasa-parsed", default=None,
                     help="furniture/raw/cubicasa5k/parsed_furniture_calibrated.json — опционально, добавит CubiCasa5K")
    ap.add_argument("--cubicasa-root", default="furniture/raw/cubicasa5k/extracted/cubicasa5k")
    args = ap.parse_args()

    classes_cfg = yaml.safe_load(open(args.classes_yaml, encoding="utf-8"))
    fg = classes_cfg["foreground_classes"]
    name_to_idx = {name: i for i, (cid, name) in enumerate(sorted(fg.items(), key=lambda kv: int(kv[0])))}
    idx_to_name = {i: name for name, i in name_to_idx.items()}

    records = json.load(open(args.parsed, "r", encoding="utf-8"))
    out_root = Path(args.out)
    images_root = Path(args.images_root)

    n_written = 0
    # file_name в parsed_furniture.json — "train/floor_image_1.tiff" и т.п.,
    # относительно SFPI/Images/
    for rec in records:
        split = SPLIT_MAP.get(rec["split"])
        if split is None or not rec["instances"]:
            continue
        src_img = images_root / rec["file_name"]
        if not src_img.is_file():
            continue

        img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = rec["height"], rec["width"]

        img_out_dir = out_root / "images" / split
        lbl_out_dir = out_root / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(rec["file_name"]).stem
        jpg_path = img_out_dir / f"{stem}.jpg"
        cv2.imwrite(str(jpg_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        lines = []
        for inst in rec["instances"]:
            cls_idx = name_to_idx.get(inst["class_name"])
            if cls_idx is None:
                continue
            pts = np.array(inst["polygon"], dtype=np.float64).reshape(-1, 2)
            pts[:, 0] /= w
            pts[:, 1] /= h
            pts = np.clip(pts, 0.0, 1.0)
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{cls_idx} {coords}")

        if not lines:
            jpg_path.unlink(missing_ok=True)
            continue
        (lbl_out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_written += 1

    if args.cubicasa_parsed:
        add_cubicasa(out_root, Path(args.cubicasa_parsed), Path(args.cubicasa_root),
                     name_to_idx, classes_cfg["cubicasa5k_to_canonical"],
                     classes_cfg["split_seed"], classes_cfg["train_val_split"])

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: idx_to_name[i] for i in sorted(idx_to_name)},
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"[build_furniture_yolo] написано {n_written} изображений -> {out_root}")
    print(f"[build_furniture_yolo] классы: {idx_to_name}")


if __name__ == "__main__":
    main()
