"""
Визуализация OCR-детекций на UGC-планах + эвристика поиска площади: по описанию
пользователя, в центре комнаты обычно печатают ДВА числа друг под другом
(сверху — номер комнаты, маленькое целое; снизу — площадь, десятичное число).
Ищем такие вертикальные пары: если над десятичным числом стоит маленькое целое
с похожим x-центром и небольшим зазором по y — нижнее число помечаем как
кандидата в площадь.

Ничего не обучаем — только эвристика поверх готового OCR для быстрой проверки
на глаз.

Запуск:
    python furniture/data_prep/ocr_visualize.py --n 5 --out-dir /tmp/ocr_vis
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np

INT_RE = re.compile(r"^\d{1,2}$")
DECIMAL_RE = re.compile(r"^\d{1,2}[.,]\d{1,2}$")
HEIGHT_RE = re.compile(r"^h\s*=", re.IGNORECASE)


def bbox_to_xyxy(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def find_area_candidates(results):
    """results: list of (bbox, text, conf). Возвращает set индексов — кандидаты в площадь."""
    boxes = []
    for bbox, text, conf in results:
        x0, y0, x1, y1 = bbox_to_xyxy(bbox)
        boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                      "cx": (x0 + x1) / 2, "h": y1 - y0, "text": text.strip()})

    area_idx = set()
    for i, a in enumerate(boxes):
        if not INT_RE.match(a["text"]):
            continue
        for j, b in enumerate(boxes):
            if i == j:
                continue
            b_text = b["text"].replace(",", ".").strip()
            if HEIGHT_RE.match(b_text) or not DECIMAL_RE.match(b_text):
                continue
            # b должен быть прямо под a (в разумных пределах — OCR-боксы часто
            # немного повёрнуты/смещены на реальных фото): широкий допуск по X
            # и Y относительно среднего размера бокса, а не жёсткий порог.
            avg_h = (a["h"] + b["h"]) / 2
            gap = b["y0"] - a["y1"]
            if -avg_h <= gap <= avg_h * 3 and abs(a["cx"] - b["cx"]) < avg_h * 4:
                area_idx.add(j)
    return area_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/ugc_test/images")
    ap.add_argument("--out-dir", default="furniture/raw/ocr_vis")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(["ru", "en"], gpu=True)
    img_dir = Path(args.images_dir)
    files = sorted(img_dir.iterdir())[: args.n]

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        results = reader.readtext(img)
        area_idx = find_area_candidates(results)

        vis = img.copy()
        for i, (bbox, text, conf) in enumerate(results):
            pts = np.array(bbox, dtype=np.int32)
            is_area = i in area_idx
            color = (0, 0, 255) if is_area else (0, 200, 0)  # BGR: красный=площадь, зелёный=прочее
            thickness = 3 if is_area else 1
            cv2.polylines(vis, [pts], True, color, thickness)
            label = f"{text} ({conf:.2f})" + (" AREA?" if is_area else "")
            org = (int(pts[:, 0].min()), max(0, int(pts[:, 1].min()) - 5))
            cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_path = out_dir / f"{f.stem}_ocr.png"
        cv2.imwrite(str(out_path), vis)
        print(f"[ocr_visualize] {f.name}: {len(results)} detections, "
              f"{len(area_idx)} area-candidates -> {out_path}")


if __name__ == "__main__":
    main()
