"""
Растеризует GT-полигоны data/ugc_test/test_coco.json в семантические PNG-маски
(1 канал, значение пикселя = category_id, 0 = фон) — для тех случаев, когда
нужен именно файл маски, а не COCO JSON (визуализация, сторонние инструменты,
ручная проверка). Порядок отрисовки — по возрастанию площади инстанса (сначала
крупные комнаты, потом стены/проёмы поверх них), чтобы мелкие классы не
перекрывались крупными при наложении.

Запуск:
    python data_prep/make_ugc_semantic_masks.py
Результат:
    data/ugc_test/semantic_masks/<file_name>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_utils import load_paths  # noqa: E402


def ann_to_binary_mask(ann: dict, h: int, w: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    rles = mask_utils.frPyObjects(seg, h, w)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(bool)


def main():
    paths = load_paths()
    ugc_test_dir = Path(paths["derived"]["ugc_test_dir"])
    gt_path = ugc_test_dir / "test_coco.json"
    out_dir = ugc_test_dir / "semantic_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(gt_path))
    img_ids = coco.getImgIds()

    n_written = 0
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        anns_sorted = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_sorted:
            bin_mask = ann_to_binary_mask(ann, h, w)
            mask[bin_mask] = ann["category_id"]

        out_name = Path(info["file_name"]).with_suffix(".png").name
        cv2.imwrite(str(out_dir / out_name), mask)
        n_written += 1

    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    mapping_path = out_dir / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"id_to_name": cats, "background_id": 0}, f, ensure_ascii=False, indent=2)

    print(f"[make_ugc_semantic_masks] написано {n_written} масок -> {out_dir}")
    print(f"[make_ugc_semantic_masks] маппинг классов -> {mapping_path}")


if __name__ == "__main__":
    main()
