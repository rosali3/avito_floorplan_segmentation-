"""
SFPI уже в COCO-формате (Annotations/{train,val,test}_annotation.json внутри
вложенного SFPI.zip) — парсер не нужен, только фильтрация категорий (door/
window выкидываем, они уже покрыты opening основной модели) и маппинг на
канонические имена (furniture_classes.yaml: sfpi_to_canonical).

Выход — тот же формат, что у parse_cubicasa_svg.py ([{"file_name","split",
"width","height","instances":[{"class_name","polygon"}]}]), чтобы оба
источника было легко слить в один датасет одним и тем же мерджером.

Запуск:
    python furniture/data_prep/parse_sfpi.py \
        --sfpi-zip furniture/raw/sfpi/"Floor plan dataset"/SFPI.zip \
        --out furniture/raw/sfpi/parsed_furniture.json
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfpi-zip", required=True)
    ap.add_argument("--classes-yaml", default="furniture/configs/furniture_classes.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    classes_cfg = yaml.safe_load(open(args.classes_yaml, encoding="utf-8"))
    sfpi_to_canonical = classes_cfg["sfpi_to_canonical"]

    z = zipfile.ZipFile(args.sfpi_zip)
    results = []
    n_skipped_cat = Counter()

    for split in SPLITS:
        with z.open(f"SFPI/Annotations/{split}_annotation.json") as f:
            coco = json.load(f)
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        images_by_id = {im["id"]: im for im in coco["images"]}

        instances_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            raw_name = cat_id_to_name[ann["category_id"]]
            canon = sfpi_to_canonical.get(raw_name)
            if canon is None:
                n_skipped_cat[raw_name] += 1
                continue
            seg = ann["segmentation"]
            poly = seg[0] if isinstance(seg, list) and seg and isinstance(seg[0], list) else seg
            instances_by_image.setdefault(ann["image_id"], []).append(
                {"class_name": canon, "polygon": poly}
            )

        for img_id, instances in instances_by_image.items():
            img = images_by_id[img_id]
            results.append({
                "file_name": f"{split}/{img['file_name']}",
                "split": split,
                "width": img["width"],
                "height": img["height"],
                "instances": instances,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    total_inst = sum(len(r["instances"]) for r in results)
    print(f"[parse_sfpi] {len(results)} планов, {total_inst} инстансов мебели -> {args.out}")

    c = Counter(inst["class_name"] for r in results for inst in r["instances"])
    print("[parse_sfpi] распределение классов:", dict(c.most_common()))
    print("[parse_sfpi] пропущено (door/window, не мебель):", dict(n_skipped_cat))


if __name__ == "__main__":
    main()
