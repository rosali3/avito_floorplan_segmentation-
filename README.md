# Сравнительное исследование instance segmentation моделей (floor-plan данные)

Сравниваются: **RF-DETR-Seg (Medium)**, **YOLO-seg** (YOLO11/YOLOv8, ultralytics),
**Mask R-CNN** (MMDetection, классический бейзлайн), **SegFormer** (семантическая
сегментация + постобработка в instances) и **SAM** (zero-shot через Grounded-SAM,
и дообученный decoder).

Эта папка **самодостаточна и ничего не пишет за своими пределами**:
- `combined_out/` (train/valid источник) и `ugc_labeled/` (test источник) —
  только читаются;
- все производные данные, конфиги, чекпоинты, логи и предсказания — в
  `data/` и `output/` внутри ЭТОЙ папки.

**Ничего в этой сессии не обучалось и не запускалось** — только подготовлен
код. Все команды ниже нужно выполнить самостоятельно. См. `progress.md` —
там разбор данных, обоснование решений и то, что стоит перепроверить перед
первым реальным запуском (особенно версии-специфичные API rfdetr/mmdet).

## 0. Общее окружение

Рекомендуется **отдельный venv на каждую модель** — библиотеки (rfdetr,
ultralytics, mmdetection, transformers, segment-anything+groundingdino) тянут
конфликтующие версии torch/mmcv/numpy. Общие для data_prep/eval пакеты:

```bash
pip install pyyaml opencv-python-headless numpy tqdm pycocotools
```

Локальная GPU на момент подготовки — **RTX 3080 8GB** (`nvidia-smi` проверено).
Дефолтные batch-size/grad-accum во всех train.py подобраны под неё; если
реально обучаешь на другой карте — просто передай другие значения флагами
(см. `--help` каждого скрипта), никаких хардкодов, требующих правки кода, нет
(кроме отдельных мест, явно помеченных в progress.md).

## 1. Подготовка данных — конвертер (один раз, для всех моделей)

**Данные ещё НЕ конвертированы** — ни один из скриптов ниже пока не
запускался (см. `progress.md`). Это первое, что нужно сделать перед любым
обучением.

Самый простой способ — один скрипт, который прогоняет весь конвертер по
порядку:

```bash
pip install -r requirements-common.txt

# Linux (в т.ч. будущий GPU-сервер) / Git Bash на Windows:
bash run_data_prep.sh

# нативный Windows PowerShell:
powershell -ExecutionPolicy Bypass -File run_data_prep.ps1
```

Либо теми же 4 шагами вручную (из корня `claude_instseg_compare/`):

```bash
# 1. train_coco.json / valid_coco.json + train_files.txt / valid_files.txt
#    (instance-аннотации из семантических масок combined_out, свой 80/20 сплит)
python data_prep/build_train_val_coco.py
#   быстрая проверка на подмножестве перед полным прогоном (~20к картинок):
#   python data_prep/build_train_val_coco.py --limit 200

# 2. объединённый test-сплит из ugc_labeled (train+valid+test roboflow-папки),
#    категории сведены к нашей таксономии (restroom->bathroom и т.д.)
python data_prep/prepare_ugc_test.py

# 3. форматы под конкретные модели (запускать по мере необходимости):
python data_prep/coco_to_yolo_seg.py        # для models/yolo_seg
python data_prep/prepare_rfdetr_dataset.py  # для models/rfdetr_seg
# Mask R-CNN (MMDetection) и SegFormer читают train_coco.json/valid_coco.json
# напрямую, отдельный конвертер не нужен.
```

### Перенос конвертера на сервер (когда появится доступ)

Конвертер не хардкодит Windows-пути в коде — ВСЕ пути идут из
`configs/paths.yaml`. При переносе на сервер (или любую другую машину):

1. Скопируй `combined_out/`, `ugc_labeled/` и `claude_instseg_compare/` на
   сервер (в любые Linux-пути, не обязательно повторять локальную структуру).
2. Поправь 3 строки в `configs/paths.yaml` (`combined_out_root`,
   `ugc_labeled_root`, `project_root`) под новые пути — **или**, если не
   хочешь трогать файл, задай переменные окружения перед запуском (удобно
   для автоматизации/CI):
   ```bash
   export CLAUDE_COMBINED_OUT_ROOT=/home/User24/2907/resplan_coco
   export CLAUDE_UGC_LABELED_ROOT=/home/User24/2907/ugc_labeled
   export CLAUDE_PROJECT_ROOT=/home/User24/3007/claude_instseg_compare
   ```
3. `bash run_data_prep.sh` — тот же конвертер, без единой правки кода.

Это касается и `models/maskrcnn_mmdet/config_maskrcnn.py` — он тоже не
содержит абсолютных путей, `train.py` подставляет их программно из тех же
`paths.yaml`/env-переменных.

## 2. Обучение + инференс по каждой модели

Каждая модель имеет `train.py` (обучение, чекпоинты + CSV/TensorBoard логи в
`output/<model>/`) и `infer_and_eval*.py` (инференс на `data/ugc_test/` +
метрики через общий `eval/coco_eval_common.py`, результат в
`output/<model>/predictions/`).

| Модель | Установка | Обучение | Инференс+метрики |
|---|---|---|---|
| RF-DETR-Seg Medium | `pip install -r models/rfdetr_seg/requirements.txt` | `python models/rfdetr_seg/train.py` | `python models/rfdetr_seg/infer_and_eval.py --checkpoint output/rfdetr_seg/checkpoints/best.pth` |
| YOLO-seg | `pip install -r models/yolo_seg/requirements.txt` | `python models/yolo_seg/train.py` | `python models/yolo_seg/infer_and_eval.py --weights output/yolo_seg/checkpoints/best.pt` |
| Mask R-CNN (MMDetection) | см. `models/maskrcnn_mmdet/SETUP.md` (отдельный venv, mim) | `python models/maskrcnn_mmdet/train.py --load-from base_config/<чекпоинт>.pth` | `python models/maskrcnn_mmdet/infer_and_eval.py --checkpoint output/maskrcnn_mmdet/checkpoints/best.pth` |
| SegFormer | `pip install -r models/segformer/requirements.txt` | `python models/segformer/train.py` | `python models/segformer/infer_and_eval.py --checkpoint output/segformer/checkpoints/best.pt` |
| SAM zero-shot | см. `models/sam/SETUP.md` (+ GroundingDINO) | не требуется | `python models/sam/zero_shot_grounded_sam.py --gdino-config ... --gdino-checkpoint ... --sam-checkpoint ...` |
| SAM fine-tuned | см. `models/sam/SETUP.md` | `python models/sam/finetune_sam.py --sam-checkpoint ...` | `python models/sam/infer_and_eval_finetuned.py --gdino-config ... --gdino-checkpoint ... --sam-checkpoint ... --decoder-checkpoint output/sam_finetuned/checkpoints/best.pt` |

Каждый `train.py` поддерживает `--help` со всеми гиперпараметрами (epochs,
batch-size, lr, ...) — дефолты подобраны под 8GB VRAM, см. комментарии в
начале каждого файла.

## 3. Итоговая таблица + отчёт

После того как хотя бы часть `infer_and_eval*.py` отработала (не обязательно
все сразу — скрипт просто пометит недостающие модели как "NOT RUN"):

```bash
python eval/aggregate_results.py
```

Результат: `output/final_results.csv`, `output/final_results.md` (общая
таблица mAP50/mAP50-95/mask AP/AR100 + разбивка по классам). Шаблон для
финального письменного отчёта с выводами — `docs/report_template.md`
(заполни числами после прогона, там же — куда смотреть в первую очередь:
маленький test-сплит, разная методология SegFormer/SAM vs честных
детекторов, и т.д.).

## Структура

```
configs/            classes.yaml (таксономия+маппинг), paths.yaml (все пути)
run_data_prep.sh/.ps1 конвертер одной командой (см. "Перенос на сервер" выше)
data_prep/           коллекция скриптов подготовки данных (шаг 1 выше)
eval/               coco_eval_common.py (общий COCOeval), aggregate_results.py
models/<name>/       train.py, infer_and_eval.py, requirements/SETUP по модели
data/                (создаётся скриптами) train_coco.json, ugc_test/, yolo_ds/, rfdetr_ds/
output/<model>/      (создаётся train.py) checkpoints/, logs/, predictions/
docs/report_template.md
progress.md          журнал решений/допущений/известных проблем — читай первым
```
