"""
Строит train_coco.json / valid_coco.json (instance segmentation) из семантических
масок combined_out (ResPlan_v2 + CubiCasa5K), 8-классовая таксономия -> 7
foreground классов (background=0 не является instance-классом).

ТОЛЬКО ЧИТАЕТ combined_out/. Пишет исключительно в project_root/data/.

Собственный 80/20 сплит (seed=42, стратифицированный по источнику resplan/cubicasa),
независимый от train.txt/val_cubicasa.txt/test_cubicasa.txt из combined_out — те
решают другую задачу (domain-transfer ablation), см. configs/classes.yaml.

Запуск:
    python data_prep/build_train_val_coco.py
    # быстрая проверка на подмножестве:
    python data_prep/build_train_val_coco.py --limit 200

Результат:
    data/train_coco.json, data/valid_coco.json   (file_name = "resplan/14877.png" и т.п.,
                                                    относительно combined_out/images/)
    data/train_files.txt, data/valid_files.txt    (тот же список, по одному файлу на
                                                    строку — формат, совместимый с
                                                    resplan_dataset.ResPlanSegmentation(ids=...),
                                                    используется моделью SegFormer)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm

from coco_utils import (
    CocoBuilder,
    canonical_categories,
    load_classes,
    load_paths,
    mask_to_instances,
)

SOURCES = ("resplan", "cubicasa")


def list_images(combined_out_root: Path) -> dict[str, list[str]]:
    """{"resplan": ["14877.png", ...], "cubicasa": [...]}"""
    out = {}
    for src in SOURCES:
        d = combined_out_root / "images" / src
        files = sorted(p.name for p in d.glob("*.png"))
        if not files:
            raise RuntimeError(f"не нашёл ни одной картинки в {d}")
        out[src] = files
    return out


def stratified_split(files_by_source: dict[str, list[str]], train_frac: float, seed: int):
    rng = random.Random(seed)
    train, valid = [], []
    for src, files in files_by_source.items():
        files = files[:]
        rng.shuffle(files)
        n_train = round(len(files) * train_frac)
        train.extend(f"{src}/{name}" for name in files[:n_train])
        valid.extend(f"{src}/{name}" for name in files[n_train:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def build_split_coco(
    rel_paths: list[str],
    combined_out_root: Path,
    categories: list[dict],
    classes_cfg: dict,
    dilate_gap_px: int = 0,
) -> CocoBuilder:
    fg_ids = [c["id"] for c in categories]
    builder = CocoBuilder(categories)
    n_no_instances = 0

    for rel in tqdm(rel_paths, desc="extracting instances"):
        mask_path = combined_out_root / "masks" / rel
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"не читается маска: {mask_path}")
        if mask.ndim != 2:
            mask = mask[:, :, 0]
        h, w = mask.shape[:2]

        instances = mask_to_instances(
            mask,
            foreground_class_ids=fg_ids,
            min_area_px=classes_cfg["min_instance_area_px"],
            approx_epsilon_frac=classes_cfg["polygon_approx_epsilon_frac"],
            ignore_id=classes_cfg["ignore_id"],
            dilate_gap_px=dilate_gap_px,
        )
        if not instances:
            n_no_instances += 1
            continue

        image_id = builder.add_image(file_name=rel, width=w, height=h)
        for inst in instances:
            builder.add_annotation(
                image_id=image_id,
                category_id=inst["category_id"],
                bbox=inst["bbox"],
                area=inst["area"],
                segmentation=inst["segmentation"],
                iscrowd=inst["iscrowd"],
            )

    if n_no_instances:
        print(f"[build_train_val_coco] предупреждение: {n_no_instances} картинок без "
              f"единого instance (маска пустая/только background+ignore) — включены "
              f"в images, но без annotations.")
    return builder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                     help="ограничить число картинок на источник (для быстрой проверки)")
    ap.add_argument("--dilate-gap-px", type=int, default=None,
                     help="слить фрагменты одного класса, разделённые ignore-полосой до N px "
                          "(см. docstring mask_to_instances в coco_utils.py). "
                          "По умолчанию берётся classes.yaml:dilate_gap_px.")
    ap.add_argument("--out-dir", default=None,
                     help="куда писать train_coco.json/valid_coco.json — по умолчанию "
                          "paths.yaml:derived.data_dir. Укажи отдельную папку, чтобы не "
                          "перезаписать данные, которые уже читает работающее обучение.")
    args = ap.parse_args()

    paths = load_paths()
    classes_cfg = load_classes()
    combined_out_root = Path(paths["combined_out_root"])
    data_dir = Path(args.out_dir) if args.out_dir else Path(paths["derived"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    dilate_gap_px = args.dilate_gap_px if args.dilate_gap_px is not None else classes_cfg.get("dilate_gap_px", 0)
    print(f"[build_train_val_coco] dilate_gap_px = {dilate_gap_px}")

    print(f"[build_train_val_coco] combined_out_root = {combined_out_root}")
    files_by_source = list_images(combined_out_root)
    for src, files in files_by_source.items():
        print(f"  source={src}: {len(files)} images")
        if args.limit:
            files_by_source[src] = files[: args.limit]

    train_rel, valid_rel = stratified_split(
        files_by_source,
        train_frac=classes_cfg["train_val_split"],
        seed=classes_cfg["split_seed"],
    )
    print(f"[build_train_val_coco] split: train={len(train_rel)} valid={len(valid_rel)} "
          f"(seed={classes_cfg['split_seed']}, frac={classes_cfg['train_val_split']})")

    categories = canonical_categories(classes_cfg)

    for split_name, rel_paths in (("train", train_rel), ("valid", valid_rel)):
        builder = build_split_coco(rel_paths, combined_out_root, categories, classes_cfg,
                                    dilate_gap_px=dilate_gap_px)
        builder.save(data_dir / f"{split_name}_coco.json")
        with open(data_dir / f"{split_name}_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(rel_paths) + "\n")

    print("[build_train_val_coco] done. file_name внутри JSON — относительный путь "
          f"внутри {combined_out_root / 'images'}.")


if __name__ == "__main__":
    main()
