# Mask R-CNN (MMDetection) — установка и запуск

MMDetection версий и связки torch/mmcv/mmdet — самая хрупкая часть всего
исследования. Рекомендуется **отдельный venv** только для этой модели.

## 1. Установка (venv отдельно от остальных моделей)

```bash
python -m venv .venv-mmdet
.venv-mmdet\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmengine "mmcv>=2.0.0,<2.2.0"
mim install mmdetection
```

Если `mim install mmdetection` не подтягивает нужную версию — альтернатива,
клонировать репозиторий и поставить в editable-режиме:
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -v -e .
```

## 2. Скачать базовый конфиг + COCO-претрейн чекпоинт (transfer learning)

Из папки `models/maskrcnn_mmdet/`:
```bash
mim download mmdet --config mask_rcnn_r50_fpn_1x_coco --dest base_config
```
Это положит `base_config/mask_rcnn_r50_fpn_1x_coco.py` и
`base_config/mask_rcnn_r50_fpn_1x_coco_*.pth` — наш `config_maskrcnn.py`
наследуется от первого (`_base_`) и дообучается (`load_from`) со второго.

## 3. Подготовить данные (из корня claude_instseg_compare/)

```bash
python data_prep/build_train_val_coco.py
```
(MMDetection читает COCO JSON напрямую — отдельный конвертер, в отличие от
YOLO/rfdetr, не нужен; только сами train_coco.json/valid_coco.json.)

## 4. Обучение

Найди точное имя скачанного чекпоинта (`ls base_config/*.pth`) и передай его
через `--load-from` — имя содержит хэш, который заранее не известен:

```bash
python models/maskrcnn_mmdet/train.py --load-from base_config/mask_rcnn_r50_fpn_1x_coco_<хэш>.pth
python models/maskrcnn_mmdet/train.py --epochs 36 --batch-size 4 --load-from base_config/...pth
```

Все пути и список классов train.py подставляет сам из `configs/paths.yaml` /
`configs/classes.yaml` — `config_maskrcnn.py` абсолютных путей не содержит,
поэтому при переносе на сервер достаточно поправить `configs/paths.yaml`
(или выставить `CLAUDE_COMBINED_OUT_ROOT`/`CLAUDE_PROJECT_ROOT`, см. корневой README).

## 5. Инференс + метрики на ugc test

```bash
python data_prep/prepare_ugc_test.py   # если ещё не запускал
python models/maskrcnn_mmdet/infer_and_eval.py \
    --checkpoint output/maskrcnn_mmdet/checkpoints/best.pth
```

## Логи

`train.py` включает `TensorboardVisBackend` (см. `output/maskrcnn_mmdet/checkpoints/vis_data/`)
и стандартный текстовый/json-лог mmengine (`output/maskrcnn_mmdet/checkpoints/*.log.json` —
per-iteration loss, per-epoch val bbox/segm mAP). Лучший чекпоинт по `coco/segm_mAP`
сохраняется автоматически (`best_coco_segm_mAP_epoch_*.pth`), финальный — `epoch_<last>.pth`;
`train.py` копирует их в `best.pth`/`final.pth` для единообразия с другими моделями.
