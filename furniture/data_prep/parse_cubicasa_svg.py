"""
Самодостаточный парсер CubiCasa5K SVG-аннотаций — НЕ зависит от их тяжёлого
`floortrans` пакета (там torch/opencv под их собственный тензорный пайплайн),
только `svgelements` (чистый Python, сам разворачивает вложенные transform).

Формат разметки CubiCasa5K (см. floortrans/loaders/house.py):
  - комната:  <... class="Space <RoomType> ..."> — не используем здесь
    (комнаты нам не нужны, у нас уже есть своя room-taxonomy в основном проекте)
  - мебель:   <... class="FixedFurniture <IconName> ...">
  - дверь/окно: <... id="Door"> / <... id="Window"> — не используем здесь
    (opening уже покрыт основной instance-seg моделью проекта)

Извлекаем ТОЛЬКО FixedFurniture-элементы (мебель/сантехника).

Запуск:
    python furniture/data_prep/parse_cubicasa_svg.py \
        --cubicasa-root furniture/raw/cubicasa5k/extracted/cubicasa5k \
        --out furniture/raw/cubicasa5k/parsed_furniture.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from svgelements import SVG, Path as SvgPath, Shape

N_SAMPLE_POINTS = 24  # аппроксимация кривых полигоном


def element_to_polygon(el) -> list[float] | None:
    """Абсолютные (после всех transform) координаты полигона, [x1,y1,x2,y2,...].

    class="FixedFurniture <Name> ..." висит на <g> (Group), а не на самой
    фигуре — у Group нет пути для сэмплирования точек, зато Group.bbox()
    уже учитывает все вложенные transform, поэтому для групп берём просто
    прямоугольник по bbox; для отдельных Shape (если такое встретится)
    сэмплируем реальный контур.
    """
    if isinstance(el, Shape):
        path = SvgPath(el)
        if len(path) == 0:
            return None
        pts = []
        for i in range(N_SAMPLE_POINTS + 1):
            t = i / N_SAMPLE_POINTS
            pt = path.npoint(t)
            if pt is None:
                continue
            pts.extend([float(pt[0]), float(pt[1])])
        if len(pts) < 6:
            return None
        return pts

    bbox = el.bbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
        return None
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def parse_one_svg(svg_path: Path) -> tuple[int, int, list[dict]]:
    """Возвращает (width, height, [{"class_name": str, "polygon": [x,y,...]}])."""
    svg = SVG.parse(str(svg_path))
    width = int(float(svg.values.get("width", 0)) or 0)
    height = int(float(svg.values.get("height", 0)) or 0)

    instances = []
    for el in svg.elements():
        cls_attr = el.values.get("class", "") if hasattr(el, "values") else ""
        if "FixedFurniture " in cls_attr:
            name = cls_attr.replace("FixedFurniture ", "").split(" ")[0]
            poly = element_to_polygon(el)
            if poly:
                instances.append({"class_name": name, "polygon": poly})
    return width, height, instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubicasa-root", required=True,
                     help="папка с подпапками high_quality/high_quality_architectural/colorful, "
                          "внутри каждой — папки <id> с model.svg + F1_scaled.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.cubicasa_root)
    svg_files = sorted(root.rglob("model.svg"))
    if args.limit:
        svg_files = svg_files[: args.limit]
    print(f"[parse_cubicasa_svg] найдено {len(svg_files)} SVG-файлов под {root}")

    results = []
    n_errors = 0
    for svg_path in svg_files:
        try:
            width, height, instances = parse_one_svg(svg_path)
        except Exception as e:
            n_errors += 1
            continue
        if not instances:
            continue
        # изображение обычно рядом, F1_scaled.png или похожее — сохраним относительный путь папки
        results.append({
            "svg_path": str(svg_path.relative_to(root)),
            "folder": str(svg_path.parent.relative_to(root)),
            "width": width,
            "height": height,
            "instances": instances,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    total_inst = sum(len(r["instances"]) for r in results)
    print(f"[parse_cubicasa_svg] распарсено {len(results)} планов, "
          f"{total_inst} инстансов мебели, ошибок парсинга: {n_errors}")
    print(f"[parse_cubicasa_svg] -> {args.out}")

    from collections import Counter
    c = Counter(inst["class_name"] for r in results for inst in r["instances"])
    print("[parse_cubicasa_svg] распределение классов:", dict(c.most_common()))


if __name__ == "__main__":
    main()
