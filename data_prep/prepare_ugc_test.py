"""
Собирает единый TEST-сплит из ugc_labeled/{train,valid,test} (Roboflow COCO export).

ВАЖНО: несмотря на имена подпапок Roboflow (train/valid/test), по требованию
задачи ВЕСЬ ugc_labeled используется только как test — не для обучения/валидации
моделей. Здесь мы просто объединяем все три подпапки в один test-сплит, чтобы
получить чуть больше test-картинок (датасет крошечный, ~33 картинки суммарно).

Категории Roboflow сведены к нашей train-таксономии (configs/classes.yaml):
  - restroom -> bathroom (см. classes.yaml, явное указание из ТЗ)
  - door/window/enterence -> opening
  - bathroom/kitchen/balcony/wall -> как есть
  - toilet(id root)/coridor/hall/room/stairs/storage -> ИСКЛЮЧЕНЫ из GT как классы,
    но их геометрия сохраняется в top-level ключе "ignore_regions" (не как annotations!) —
    чтобы модель не штрафовалась как false positive за живой/спальню и т.п. именно
    в этой зоне (мы не знаем истинный тип "room", поэтому такое предсказание не
    может быть ни правильным, ни неправильным). См. eval/coco_eval_common.py.

ТОЛЬКО ЧИТАЕТ ugc_labeled/. Пишет в project_root/data/ugc_test/
(картинки копируются, не симлинкаются — чтобы test-сплит был самодостаточным
и не ломался, если исходная папка переедет/удалится).

Запуск:
    python data_prep/prepare_ugc_test.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from coco_utils import canonical_categories, load_classes, load_paths

SPLITS = ("train", "valid", "test")  # исходные roboflow-подпапки, все идут в наш test


def build_category_remap(raw_categories: list[dict], classes_cfg: dict) -> tuple[dict[int, int], list[str]]:
    """raw category id -> canonical category id. Возвращает также список
    исключённых имён категорий, реально встреченных в файле (для лога)."""
    name_to_canon_id = {}
    fg = classes_cfg["foreground_classes"]
    name_to_id = {name: cid for cid, name in fg.items()}
    ugc_map = classes_cfg["ugc_to_canonical"]
    excluded = set(classes_cfg["ugc_excluded_categories"])

    raw_id_to_canon_id: dict[int, int] = {}
    excluded_seen = []
    for cat in raw_categories:
        raw_name = cat["name"]
        raw_id = cat["id"]
        if raw_name in ugc_map:
            canon_name = ugc_map[raw_name]
            raw_id_to_canon_id[raw_id] = name_to_id[canon_name]
        elif raw_name in excluded:
            excluded_seen.append(raw_name)
            continue
        else:
            raise ValueError(
                f"Категория '{raw_name}' (id={raw_id}) из ugc_labeled не описана ни в "
                f"ugc_to_canonical, ни в ugc_excluded_categories в configs/classes.yaml. "
                f"Добавь её явно в один из списков, прежде чем продолжать."
            )
    return raw_id_to_canon_id, excluded_seen


def main():
    paths = load_paths()
    classes_cfg = load_classes()
    ugc_root = Path(paths["ugc_labeled_root"])
    out_dir = Path(paths["derived"]["ugc_test_dir"])
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    categories = canonical_categories(classes_cfg)
    merged_images = []
    merged_anns = []
    merged_ignore_regions = []
    next_image_id = 1
    next_ann_id = 1

    n_dropped_by_excluded_cat = 0
    n_copied = 0

    for split in SPLITS:
        split_dir = ugc_root / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.is_file():
            print(f"[prepare_ugc_test] пропускаю {split}: нет {ann_path}")
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        raw_id_to_canon_id, excluded_seen = build_category_remap(coco["categories"], classes_cfg)
        if excluded_seen:
            print(f"[prepare_ugc_test] {split}: категории исключены из GT: {sorted(set(excluded_seen))}")

        old_to_new_image_id = {}
        for img in coco["images"]:
            src_path = split_dir / img["file_name"]
            if not src_path.is_file():
                raise FileNotFoundError(f"нет файла картинки: {src_path}")
            # префикс сплитом на случай совпадения имён между train/valid/test
            new_name = f"{split}__{img['file_name']}"
            dst_path = out_images_dir / new_name
            shutil.copyfile(src_path, dst_path)
            n_copied += 1

            new_id = next_image_id
            next_image_id += 1
            old_to_new_image_id[img["id"]] = new_id
            merged_images.append({
                "id": new_id,
                "file_name": new_name,
                "width": img["width"],
                "height": img["height"],
            })

        for ann in coco["annotations"]:
            raw_cat_id = ann["category_id"]
            if raw_cat_id not in raw_id_to_canon_id:
                n_dropped_by_excluded_cat += 1
                merged_ignore_regions.append({
                    "image_id": old_to_new_image_id[ann["image_id"]],
                    "segmentation": ann["segmentation"],
                    "bbox": ann["bbox"],
                })
                continue
            merged_anns.append({
                "id": next_ann_id,
                "image_id": old_to_new_image_id[ann["image_id"]],
                "category_id": raw_id_to_canon_id[raw_cat_id],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "segmentation": ann["segmentation"],
                "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1

    merged = {
        "info": {"description": "ugc_labeled merged test split (train+valid+test roboflow "
                                 "folders combined; categories remapped per configs/classes.yaml)"},
        "licenses": [],
        "images": merged_images,
        "annotations": merged_anns,
        "categories": categories,
        "ignore_regions": merged_ignore_regions,
    }
    out_json = out_dir / "test_coco.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"[prepare_ugc_test] скопировано картинок: {n_copied}")
    print(f"[prepare_ugc_test] аннотаций сохранено как ignore_regions (исключённые категории): "
          f"{n_dropped_by_excluded_cat}")
    print(f"[prepare_ugc_test] итог: images={len(merged_images)} annotations={len(merged_anns)} "
          f"-> {out_json}")
    if len(merged_images) < 50:
        print("[prepare_ugc_test] ВНИМАНИЕ: test-сплит очень маленький (<50 картинок). "
              "mAP на нём будет статистически шумным — см. progress.md / отчёт.")


if __name__ == "__main__":
    main()
