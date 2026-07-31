"""
Общие утилиты, разделяемые всеми data_prep/* скриптами:
  - загрузка configs/classes.yaml и configs/paths.yaml
  - извлечение instance-аннотаций из семантической маски (connected components)
  - минимальный COCO-JSON билдер (images/annotations/categories)

Ничего не пишет и не читает за пределами того, что ей явно передали путём
аргументов — сам факт импорта этого модуля безопасен.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _expand_vars(value: Any, context: dict) -> Any:
    """Разворачивает ${key} внутри строк paths.yaml (только на верхнем уровне derived)."""
    if isinstance(value, str):
        for k, v in context.items():
            value = value.replace(f"${{{k}}}", str(v))
        return value
    if isinstance(value, dict):
        return {k: _expand_vars(v, context) for k, v in value.items()}
    return value


# Переносимость на другую машину (напр. на сервер, когда появится доступ):
# вместо правки configs/paths.yaml можно выставить эти переменные окружения —
# они перекрывают значения из yaml, ничего в коде менять не нужно. Полезно
# для запуска на Linux-сервере с другими путями (там же прямые /home/... пути),
# в CI, или когда путей несколько и их неудобно каждый раз редактировать руками.
_ENV_OVERRIDES = {
    "combined_out_root": "CLAUDE_COMBINED_OUT_ROOT",
    "ugc_labeled_root": "CLAUDE_UGC_LABELED_ROOT",
    "project_root": "CLAUDE_PROJECT_ROOT",
}


def load_paths(paths_yaml: str | Path | None = None) -> dict:
    path = Path(paths_yaml) if paths_yaml else PROJECT_ROOT / "configs" / "paths.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, env_name in _ENV_OVERRIDES.items():
        if os.environ.get(env_name):
            cfg[key] = os.environ[env_name]

    context = {
        "project_root": cfg["project_root"],
        "combined_out_root": cfg["combined_out_root"],
        "ugc_labeled_root": cfg["ugc_labeled_root"],
    }
    cfg["derived"] = _expand_vars(cfg["derived"], context)
    return cfg


def load_classes(classes_yaml: str | Path | None = None) -> dict:
    path = Path(classes_yaml) if classes_yaml else PROJECT_ROOT / "configs" / "classes.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Instance-экстракция из семантической маски
# ---------------------------------------------------------------------------

def mask_to_instances(
    mask: np.ndarray,
    foreground_class_ids: list[int],
    min_area_px: int = 24,
    approx_epsilon_frac: float = 0.005,
    ignore_id: int = 255,
    dilate_gap_px: int = 0,
) -> list[dict]:
    """Connected components одного класса на семантической маске = один instance.

    Возвращает список dict с ключами: category_id, bbox [x,y,w,h], area,
    segmentation (список полигонов COCO-формата [[x1,y1,x2,y2,...]]).

    dilate_gap_px: перед связыванием компонент бинарная маска класса
    дилатируется на N px (после чего каждый label обратно пересекается с
    исходной маской — халo дилатации в пиксели инстанса не попадает). Это
    сливает один "визуальный" объект, который сырые маски combined_out
    разрезают тонкой полосой ignore_id-пикселей (подтверждено: ~42% масок
    содержат ignore-пиксели), не сливая при этом реально разные объекты —
    те обычно разделены зазором шире пары пикселей. dilate_gap_px=0 —
    прежнее поведение (без слияния).

    ВАЖНО про bbox: считается напрямую из пиксельной маски компоненты
    (comp_mask), А НЕ из аппроксимированного полигона контура — polygon-
    аппроксимация (approxPolyDP) для тонких объектов (двери/окна/тонкие
    стены) может дать bbox МЕНЬШЕ истинного пиксельного экстента, из-за чего
    area (площадь в пикселях) оказывается больше площади собственного bbox —
    невозможная с точки зрения геометрии аннотация, которая портит box-таргет
    при обучении. bbox всегда точно накрывает все пиксели инстанса.
    """
    assert mask.ndim == 2, f"ожидается одноканальная маска, получено shape={mask.shape}"
    instances: list[dict] = []

    for cat_id in foreground_class_ids:
        binary = (mask == cat_id).astype(np.uint8)
        if not binary.any():
            continue

        if dilate_gap_px > 0:
            kernel = np.ones((2 * dilate_gap_px + 1, 2 * dilate_gap_px + 1), np.uint8)
            grouping = cv2.dilate(binary, kernel, iterations=1)
        else:
            grouping = binary

        num_labels, labels = cv2.connectedComponents(grouping, connectivity=8)
        for comp_id in range(1, num_labels):
            # обратно пересекаем с исходной (недилатированной) маской класса —
            # halo дилатации используется только для группировки, не для формы
            comp_mask = ((labels == comp_id) & (binary == 1)).astype(np.uint8)
            area = int(comp_mask.sum())
            if area < min_area_px:
                continue

            ys, xs = np.where(comp_mask)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            # Может быть несколько контуров на один label из-за диагональной
            # 8-связности рядом с дырками — берём все, что дают >=3 вершины.
            polygons = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 1:
                    continue
                peri = cv2.arcLength(cnt, True)
                eps = approx_epsilon_frac * peri
                approx = cv2.approxPolyDP(cnt, eps, True)
                if len(approx) < 3:
                    approx = cnt
                pts = approx.reshape(-1, 2)
                if len(pts) < 3:
                    continue
                polygons.append(pts.flatten().astype(float).tolist())

            if not polygons:
                continue

            instances.append({
                "category_id": cat_id,
                "bbox": bbox,
                "area": float(area),
                "segmentation": polygons,
                "iscrowd": 0,
            })

    return instances


# ---------------------------------------------------------------------------
# Минимальный COCO builder
# ---------------------------------------------------------------------------

class CocoBuilder:
    def __init__(self, categories: list[dict]):
        """categories: [{"id": 1, "name": "living"}, ...]"""
        self.categories = categories
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self._next_image_id = 1
        self._next_ann_id = 1

    def add_image(self, file_name: str, width: int, height: int, extra: dict | None = None) -> int:
        image_id = self._next_image_id
        self._next_image_id += 1
        rec = {"id": image_id, "file_name": file_name, "width": width, "height": height}
        if extra:
            rec.update(extra)
        self.images.append(rec)
        return image_id

    def add_annotation(self, image_id: int, category_id: int, bbox, area, segmentation, iscrowd: int = 0) -> int:
        ann_id = self._next_ann_id
        self._next_ann_id += 1
        self.annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation,
            "iscrowd": iscrowd,
        })
        return ann_id

    def to_dict(self) -> dict:
        return {
            "info": {"description": "auto-generated by claude_instseg_compare/data_prep"},
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        print(f"[coco_utils] wrote {path} "
              f"(images={len(self.images)}, annotations={len(self.annotations)}, "
              f"categories={len(self.categories)})")


def canonical_categories(classes_cfg: dict) -> list[dict]:
    """[{"id":1,"name":"living"}, ...] в порядке id, из classes.yaml foreground_classes."""
    fg = classes_cfg["foreground_classes"]
    return [{"id": int(cid), "name": name} for cid, name in sorted(fg.items(), key=lambda kv: int(kv[0]))]
