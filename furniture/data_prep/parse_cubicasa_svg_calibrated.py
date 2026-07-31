"""
v2 парсера мебели CubiCasa5K: используем ИХ калибровку (cubicasa_calib.py +
cubicasa_utils_v2.py, лежат в avito-toilet/, не редактируем — внешний код)
вместо своего SVG->PNG предположения (которое было неверным: у SVG viewBox
своя система координат ДО применения внешнего transform, room/wall-парсер
cubicasa_utils_v2 работает в тех же "сырых" координатах, что и мы — но
calibrate() достраивает точный affine SVG->растр через ECC на реальных
стенах, а не наивное масштабирование по width/height атрибуту, которое мы
пробовали раньше и получили несовпадающий aspect ratio).

Мебель извлекаем в ТЕХ ЖЕ "сырых" координатах, что и room/wall (тот же ctm-
проход по дереву, те же примитивы _parse_transform/_compose/_apply из
cubicasa_utils_v2), затем применяем ОДИН И ТОТ ЖЕ affine M к furniture-
полигонам, что calibrate() посчитал по стенам того же плана.

Запуск:
    python furniture/data_prep/parse_cubicasa_svg_calibrated.py \
        --cubicasa-root furniture/raw/cubicasa5k/extracted/cubicasa5k \
        --out furniture/raw/cubicasa5k/parsed_furniture_calibrated.json \
        --limit 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

sys.path.insert(0, r"C:\Users\user\Downloads\avito-toilet")  # cubicasa_calib.py/cubicasa_utils_v2.py — внешний код, не наш
from cubicasa_calib import calibrate  # noqa: E402
from cubicasa_utils_v2 import parse_svg, _tag, _parse_transform, _compose, _apply, _points_of  # noqa: E402

GATE = 0.70  # тот же порог качества калибровки, что в cubicasa_to_masks_v2.py


def _path_points(el) -> list[tuple[float, float]]:
    """Грубое извлечение опорных точек из <path d="...">: только M/L/A-конечные
    точки (без точной трассировки кривых) — для bbox мебели этого достаточно."""
    d = el.get("d", "")
    nums = re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", d)
    vals = [float(v) for v in nums]
    return list(zip(vals[0::2], vals[1::2]))


def _furniture_bboxes(root, ctm0) -> list[dict]:
    """Рекурсивный обход дерева, собираем bbox каждой FixedFurniture-группы
    (в "сырых" координатах, той же системе, что room/wall в cubicasa_utils_v2)."""
    out = []

    def walk(el, ctm, in_furniture, name):
        t = _tag(el)
        ctm = _compose(ctm, _parse_transform(el.get("transform")))
        cls = el.get("class", "") or ""
        if "FixedFurniture " in cls and not in_furniture:
            in_furniture = True
            name = cls.replace("FixedFurniture ", "").split(" ")[0]
            out.append({"class_name": name, "pts": []})

        if in_furniture:
            pts = []
            if t in ("polygon", "rect"):
                pts = _points_of(el)
            elif t == "path":
                pts = _path_points(el)
            elif t == "circle":
                cx, cy, r = float(el.get("cx", 0)), float(el.get("cy", 0)), float(el.get("r", 0))
                pts = [(cx - r, cy - r), (cx + r, cy + r)]
            for (x, y) in pts:
                out[-1]["pts"].append(_apply(ctm, x, y))

        for ch in el:
            walk(ch, ctm, in_furniture, name)

    walk(root, ctm0, False, None)
    return [r for r in out if len(r["pts"]) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubicasa-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root_dir = Path(args.cubicasa_root)
    svg_files = sorted(root_dir.rglob("model.svg"))
    if args.limit:
        svg_files = svg_files[: args.limit]
    print(f"[parse_calibrated] {len(svg_files)} SVG под {root_dir}")

    results = []
    n_dropped_calib, n_errors = 0, 0
    for svg_path in svg_files:
        raster_path = svg_path.parent / "F1_scaled.png"
        if not raster_path.is_file():
            continue
        try:
            floors, vb = parse_svg(str(svg_path))
            if len(floors) != 1:
                continue
            floor = floors[0]
            if len(floor.rooms) < 2 or not floor.walls:
                continue
            gray = cv2.imread(str(raster_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            cal = calibrate(floor, gray, vb)
            if cal["precision"] < GATE:
                n_dropped_calib += 1
                continue
            M = cal["M"]

            tree = ET.parse(str(svg_path))
            furn = _furniture_bboxes(tree.getroot(), (1, 0, 0, 1, 0, 0))
            if not furn:
                continue

            Rh, Rw = gray.shape
            A, t2 = M[:, :2], M[:, 2:3]
            instances = []
            for f in furn:
                pts = np.array(f["pts"], dtype=np.float64).T  # 2 x N
                tp = (A @ pts + t2).T
                x0, y0 = tp[:, 0].min(), tp[:, 1].min()
                x1, y1 = tp[:, 0].max(), tp[:, 1].max()
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(Rw, x1), min(Rh, y1)
                # санити-фильтр: грубый парсер <path> (регексом, без честного
                # разбора arc-команд) иногда даёт огромный мусорный bbox —
                # мебель физически не может занимать >35% плана по стороне
                if x1 - x0 < 2 or y1 - y0 < 2 or (x1 - x0) > 0.35 * Rw or (y1 - y0) > 0.35 * Rh:
                    continue
                poly = [x0, y0, x1, y0, x1, y1, x0, y1]
                instances.append({"class_name": f["class_name"], "polygon": poly})
            if not instances:
                continue

            results.append({
                "svg_path": str(svg_path.relative_to(root_dir)),
                "raster_path": str(raster_path.relative_to(root_dir)),
                "width": Rw, "height": Rh,
                "calib_precision": round(cal["precision"], 4),
                "instances": instances,
            })
        except Exception:
            n_errors += 1
            continue

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    total_inst = sum(len(r["instances"]) for r in results)
    print(f"[parse_calibrated] распарсено {len(results)} планов (откинуто по калибровке: "
          f"{n_dropped_calib}, ошибок: {n_errors}), {total_inst} инстансов -> {args.out}")
    c = Counter(inst["class_name"] for r in results for inst in r["instances"])
    print("[parse_calibrated] распределение классов:", dict(c.most_common()))


if __name__ == "__main__":
    main()
