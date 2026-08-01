"""
Пост-обработка бинарной маски: заливка внутренних "дыр" (мелкие вкрапления
другого класса внутри контура — текст, значки мебели, шум сегментации) и
удаление мелкого мусора-отшельника (крошечные несвязанные компоненты).

НЕ расширяет маску наружу — только заполняет уже окружённые foreground'ом
пустоты и убирает то, что заведомо шум, а не отдельная комната.

Основной кандидат на применение — семантические модели (SegFormer/UNet),
у которых predicted-маска часто "блобами" с дырами, а не сплошным контуром
комнаты, в отличие от instance-моделей с чистой mask-головой.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def fill_holes_and_denoise(mask: np.ndarray, min_area_frac: float = 0.01) -> np.ndarray:
    """mask: HxW bool. Возвращает HxW bool того же размера.
    min_area_frac — компоненты меньше этой доли ОТ САМОГО БОЛЬШОГО компонента
    выбрасываются как шум (не про реальные маленькие комнаты — про спеклы)."""
    if not mask.any():
        return mask

    filled = ndimage.binary_fill_holes(mask)

    labeled, n = ndimage.label(filled)
    if n <= 1:
        return filled
    sizes = ndimage.sum(filled, labeled, index=range(1, n + 1))
    max_size = sizes.max()
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= max_size * min_area_frac]
    out = np.isin(labeled, keep_labels)
    return out
