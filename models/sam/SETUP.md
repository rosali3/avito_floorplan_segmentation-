# SAM zero-shot (Grounded-SAM) + SAM fine-tune — установка

## 1. Пакеты

```bash
pip install segment-anything opencv-python-headless pycocotools tensorboard torch torchvision
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```
Если сборка CUDA-расширения GroundingDINO падает — она нужна не всегда;
инференс отработает и на чистом PyTorch (медленнее), см. issues репозитория.

## 2. Веса

- SAM (zero-shot, максимальное качество): `sam_vit_h_4b8939.pth`
  https://github.com/facebookresearch/segment-anything#model-checkpoints
- SAM (базовая модель для файнтюна на 8GB VRAM): `sam_vit_b_01ec64.pth`
  (vit_b — единственный реалистичный выбор для файнтюна decoder'а на локальной
  RTX 3080 8GB; vit_h/vit_l тоже можно, если карта мощнее)
- GroundingDINO: `groundingdino_swint_ogc.pth` + конфиг
  `GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` (идёт в репозитории)
  https://github.com/IDEA-Research/GroundingDINO#luggage-checkpoints

## 3. Подготовка данных

```bash
python data_prep/build_train_val_coco.py     # для finetune_sam.py (GT-боксы+маски)
python data_prep/prepare_ugc_test.py         # test для обоих вариантов SAM
```

## 4. Zero-shot (без обучения, сразу инференс+метрики)

```bash
python models/sam/zero_shot_grounded_sam.py \
    --gdino-config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino-checkpoint groundingdino_swint_ogc.pth \
    --sam-checkpoint sam_vit_h_4b8939.pth --sam-model-type vit_h
```

## 5. Fine-tune (только mask_decoder, image/prompt encoder заморожены)

```bash
python models/sam/finetune_sam.py --sam-checkpoint sam_vit_b_01ec64.pth --sam-model-type vit_b
```
Лог: `output/sam_finetuned/logs/metrics.csv` + TensorBoard в той же папке.
Чекпоинты: `output/sam_finetuned/checkpoints/{best,final}.pt` (только веса decoder'а).

## 6. Инференс + метрики дообученной модели

```bash
python models/sam/infer_and_eval_finetuned.py \
    --gdino-config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --gdino-checkpoint groundingdino_swint_ogc.pth \
    --sam-checkpoint sam_vit_b_01ec64.pth --sam-model-type vit_b \
    --decoder-checkpoint output/sam_finetuned/checkpoints/best.pt
```

## Почему боксы одинаковые в обоих вариантах SAM

И zero-shot, и finetuned вариант получают боксы от одного и того же
GroundingDINO-детектора (`models/sam/grounding_boxes.py`), а не от GT.
Так разница в итоговых метриках между двумя строчками таблицы отражает
ИСКЛЮЧИТЕЛЬНО эффект дообучения mask_decoder на нашем домене, а не разные
источники боксов — иначе сравнение "zero-shot vs finetuned" было бы нечестным
(и оба варианта остаются полностью automatic pipeline, без утечки GT, поэтому
сравнимы с остальными 4 моделями исследования "на равных").
