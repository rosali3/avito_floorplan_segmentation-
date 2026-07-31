"""
Конвертирует train_coco.json / valid_coco.json (наш формат, file_name относительно
combined_out/images/) в YOLO-seg формат, ожидаемый ultralytics:

    data/yolo_ds/
      images/train/*.png  images/valid/*.png   (hardlink на combined_out, либо copy)
      labels/train/*.txt  labels/valid/*.txt
      data.yaml

YOLO-seg строка: "<class_idx 0-based> x1 y1 x2 y2 ... xn yn" (нормализовано 0..1).
Формат поддерживает один полигон на объект — если connected component дал
несколько контуров (редкий случай дырок/диагональной связности), берём контур
с наибольшей площадью и теряем остальные. Для остальных моделей (COCO-based)
это ограничение не действует, там сохраняются все полигоны.

Запуск (после build_train_val_coco.py):
    python data_prep/coco_to_yolo_seg.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from coco_utils import load_classes, load_paths


def polygon_area(poly: list[float]) -> float:
    xs = poly[0::2]
    ys = poly[1::2]
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # NTFS hardlink, без прав администратора, без дублирования места на диске
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def convert_split(coco_path: Path, images_root: Path, out_root: Path, split: str,
                   canon_id_to_idx: dict[int, int]):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out_dir = out_root / "images" / split
    lbl_out_dir = out_root / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    n_images_written = 0
    n_instances_written = 0
    n_multi_poly_truncated = 0

    for img in coco["images"]:
        w, h = img["width"], img["height"]
        src_path = images_root / img["file_name"]
        # заменяем "resplan/14877.png" -> "resplan__14877.png", т.к. YOLO не любит
        # подпапки внутри images/train
        flat_name = img["file_name"].replace("/", "__").replace("\\", "__")
        link_or_copy(src_path, img_out_dir / flat_name)

        lines = []
        for ann in anns_by_image.get(img["id"], []):
            polys = ann["segmentation"]
            if len(polys) > 1:
                n_multi_poly_truncated += 1
                polys = [max(polys, key=polygon_area)]
            poly = polys[0]
            if len(poly) < 6:
                continue
            norm = []
            for i in range(0, len(poly), 2):
                norm.append(poly[i] / w)
                norm.append(poly[i + 1] / h)
            cls_idx = canon_id_to_idx[ann["category_id"]]
            lines.append(str(cls_idx) + " " + " ".join(f"{v:.6f}" for v in norm))
            n_instances_written += 1

        label_name = Path(flat_name).with_suffix(".txt").name
        with open(lbl_out_dir / label_name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))  # пустой файл, если инстансов нет — это ок для ultralytics
        n_images_written += 1

    print(f"[coco_to_yolo_seg] {split}: images={n_images_written} instances={n_instances_written} "
          f"multi_polygon_truncated={n_multi_poly_truncated}")


def main():
    paths = load_paths()
    classes_cfg = load_classes()
    data_dir = Path(paths["derived"]["data_dir"])
    images_root = Path(paths["combined_out_root"]) / "images"
    out_root = Path(paths["derived"]["yolo_dataset_dir"])

    fg = classes_cfg["foreground_classes"]
    ordered_ids = sorted(int(k) for k in fg.keys())
    canon_id_to_idx = {cid: i for i, cid in enumerate(ordered_ids)}
    names = [fg[cid] for cid in ordered_ids]

    convert_split(data_dir / "train_coco.json", images_root, out_root, "train", canon_id_to_idx)
    convert_split(data_dir / "valid_coco.json", images_root, out_root, "valid", canon_id_to_idx)

    data_yaml = out_root / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {out_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/valid\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names: [" + ", ".join(names) + "]\n")

    print(f"[coco_to_yolo_seg] data.yaml -> {data_yaml}")


if __name__ == "__main__":
    main()
