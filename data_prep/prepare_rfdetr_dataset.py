"""
Материализует train_coco.json / valid_coco.json в roboflow-style структуру,
которую ожидает библиотека `rfdetr` (COCO формат, автоопределяется по layout):

    data/rfdetr_ds/
      train/_annotations.coco.json + картинки
      valid/_annotations.coco.json + картинки

rfdetr сам по имени папки не разбирает train/valid/test — важно, что внутри
каждой лежит _annotations.coco.json и рядом картинки, на которые ссылается
file_name. Картинки — hardlink на combined_out (см. link_or_copy), не копия,
чтобы не занимать место повторно.

Запуск (после build_train_val_coco.py):
    python data_prep/prepare_rfdetr_dataset.py
    # из отдельной папки с train_coco.json/valid_coco.json (напр. data_v2):
    python data_prep/prepare_rfdetr_dataset.py --in-dir data_v2 --out-dir data_v2/rfdetr_ds
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from coco_utils import load_paths


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def materialize(coco_path: Path, images_root: Path, out_dir: Path):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    flat_images = []
    for img in coco["images"]:
        flat_name = img["file_name"].replace("/", "__").replace("\\", "__")
        link_or_copy(images_root / img["file_name"], out_dir / flat_name)
        flat_images.append({**img, "file_name": flat_name})

    out_coco = {**coco, "images": flat_images}
    with open(out_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(out_coco, f, ensure_ascii=False)
    print(f"[prepare_rfdetr_dataset] {out_dir}: images={len(flat_images)} "
          f"annotations={len(coco['annotations'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=None,
                     help="папка с train_coco.json/valid_coco.json — по умолчанию "
                          "paths.yaml:derived.data_dir")
    ap.add_argument("--out-dir", default=None,
                     help="куда писать roboflow-style train/valid — по умолчанию "
                          "paths.yaml:derived.rfdetr_dataset_dir")
    args = ap.parse_args()

    paths = load_paths()
    data_dir = Path(args.in_dir) if args.in_dir else Path(paths["derived"]["data_dir"])
    images_root = Path(paths["combined_out_root"]) / "images"
    out_root = Path(args.out_dir) if args.out_dir else Path(paths["derived"]["rfdetr_dataset_dir"])

    materialize(data_dir / "train_coco.json", images_root, out_root / "train")
    materialize(data_dir / "valid_coco.json", images_root, out_root / "valid")
    print(f"[prepare_rfdetr_dataset] dataset_dir для rfdetr: {out_root}")


if __name__ == "__main__":
    main()
