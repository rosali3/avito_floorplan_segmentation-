# Furniture-aware room reranker: журнал экспериментов

Отдельный под-проект (`furniture/`), не связан напрямую с основным
instance-seg сравнением моделей (RF-DETR/YOLO/Mask R-CNN/SegFormer/UNet/SAM)
— это детектор мебели/сантехники, задуманный как **реранкер типа комнаты**:
идея в том, что мебель — сильный сигнал ("не должно быть дивана в ванной",
"унитаз почти всегда значит санузел"), который можно использовать поверх
основных предсказаний room-типа для коррекции.

Никакой .md-документации по нему не было до этого файла — вся история
восстановлена из кода/конфигов/`args.yaml`/`results.csv`/логов.

## Таксономия (14 классов, `furniture/configs/furniture_classes.yaml`)

Два независимых семейства понятий из разных источников, объединённые в
общий id-маппинг:

| id | класс | источники |
|---|---|---|
| 1 | toilet | CubiCasa5K: Toilet, Urinal |
| 2 | sink | CubiCasa5K: Sink/DoubleSink/RoundSink/CornerSink/SideSink/WaterTap; FloorPlanCAD: sink; SFPI: sink1-4 |
| 3 | bathtub | CubiCasa5K: Bathtub/BathtubRound/Jacuzzi; FloorPlanCAD: bath; SFPI: tub |
| 4 | sauna_bench | CubiCasa5K: SaunaBench(Mid/High/Low) |
| 5 | fireplace | CubiCasa5K: Fireplace(Corner/Round)/PlaceForFireplace* |
| 6 | chimney | CubiCasa5K: Chimney |
| 7 | closet | CubiCasa5K: Closet/CoatCloset/ClosetTriangle/ClosetRound; FloorPlanCAD: wardrobe |
| 8 | appliance | CubiCasa5K: ElectricalAppliance/GasStove/WoodStove/SaunaStove/WashingMachine |
| 9 | sofa | FloorPlanCAD: sofa; SFPI: sofa1/sofa2 |
| 10 | bed | FloorPlanCAD: bed; SFPI: bed |
| 11 | chair | FloorPlanCAD: chair; SFPI: armchair |
| 12 | table | FloorPlanCAD: table; SFPI: table1-3 |
| 13 | cabinet | CubiCasa5K: BaseCabinet*/WallCabinet/CounterTop; FloorPlanCAD: bedside_cupboard/tv_cabinet/half_height_cabinet/high_cabinet |
| 14 | shower | CubiCasa5K: Shower/ShowerScreen*/ShowerCab/ShowerPlatform |

Явно НЕ берутся (уже покрыты основной instance-seg моделью проекта):
двери/окна (opening), стены, комнаты как таковые (room-taxonomy) — этот
детектор строго про подвижную мебель и встроенную сантехнику.

## Источники данных и как они парсились

### SFPI — основной, самый надёжный источник
Уже готовый COCO-формат (`Annotations/{train,val,test}_annotation.json`),
маппинг `sfpi_to_canonical` в classes.yaml. Использует собственный
train/val сплит SFPI (test не берётся — не нужен для обучения детектора).

### FloorPlanCAD
30 классов в источнике, берётся только 11 мебельных (`floorplancad_to_canonical`)
— остальные ~19 (двери/окна/стены/лестницы/розетки) отброшены как
структурные элементы вне задачи.

### CubiCasa5K — потребовал ДВЕ итерации парсинга SVG
1. **v1** (`parse_cubicasa_svg.py`, 31.07 11:38) — самодостаточный парсер на
   `svgelements`, извлекает `FixedFurniture <Name>` элементы. Масштабирование
   SVG→растр наивное (по width/height атрибуту) — оказалось **неверным**:
   у SVG viewBox своя система координат ДО применения внешнего transform,
   наивный подход давал несовпадающий aspect ratio → координаты мебели не
   совпадают с реальным растром. Результат — `parsed_furniture.json`
   (31.07 11:54), **не использовался для финального датасета** из-за этого
   бага (см. docstring `build_furniture_yolo.py`: "CubiCasa5K временно
   исключён из-за нерешённого расхождения координат SVG/PNG").
2. **v2 "calibrated"** (`parse_cubicasa_svg_calibrated.py`, 31.07 17:38) —
   переиспользует калибровку основного проекта (`cubicasa_calib.py` +
   `cubicasa_utils_v2.py` из `avito-toilet/`, внешний код, не редактируется):
   `calibrate()` строит точный affine SVG→растр через ECC по реальным стенам
   плана (не наивное масштабирование), затем тот же affine применяется к
   furniture-полигонам. Порог качества `GATE=0.70` (тот же, что в
   `cubicasa_to_masks_v2.py` основного проекта). Результат —
   `parsed_furniture_calibrated.json` (31.07 17:47, 5.8 МБ против 17.8 МБ
   v1 — заметно меньше прошло gate-фильтр). **Этот файл используется в
   финальном ("merged") датасете.**

### Сборка YOLO-датасета
`furniture/data_prep/build_furniture_yolo.py` — конвертирует в YOLO-seg
формат (TIFF→JPG, т.к. ultralytics не работает с TIFF из коробки),
опционально домешивает CubiCasa5K через `--cubicasa-parsed` (свой
train/val сплит: `split_seed=42`, `train_val_split=0.8`, стратифицированный
по `classes.yaml`).

**Текущий датасет** (`furniture/data/furniture_yolo_ds/`): 7900 train +
1736 val изображений. Из них train: ~900 из CubiCasa5K (`cubi_*` префикс
в имени файла), ~7000 из SFPI+FloorPlanCAD — то есть CubiCasa5K составляет
только ~11% train-выборки.

## История обучения (YOLO11n-seg, `runs/segment/furniture/output/`)

7 запусков, веса НЕ в git (`runs/` в `.gitignore`), только локально.

| Run | Данные | Модель (старт) | epochs (план/факт) | batch | mAP50(mask) | mAP50-95(mask) |
|---|---|---|---:|---:|---:|---:|
| `furniture_yolo11n` | SFPI+FloorPlanCAD (без CubiCasa) | yolo11n-seg.pt (COCO) | 50 / ? | 16 | нет results.csv | — |
| `furniture_yolo11n-2` | то же | yolo11n-seg.pt | 50 / ? | 16 | нет results.csv | — |
| `furniture_yolo11n-3` | то же | yolo11n-seg.pt | 50 / **1** | 8 | 0.964 | 0.652 |
| `furniture_yolo11n_cont` | то же | ← `-3/weights/last.pt` | 10 / **4** | 8 | **0.992** | 0.672 |
| `furniture_yolo11n_cont2` | то же | ← `_cont/weights/last.pt` | 6 / **2** | 8 | 0.992 | 0.660 |
| `furniture_yolo11n_merged` | **+ CubiCasa5K (calibrated)** | yolo11n-seg.pt (свежий старт) | 30 / **3** | 8 | 0.517 | 0.322 |
| `furniture_yolo11n_merged2` | то же | ← `_merged/weights/last.pt` | 27 / **4**, **упал** | 4 | 0.535 | 0.328 |

Первые два запуска (`furniture_yolo11n`, `-2`) не оставили `results.csv` —
похоже, упали/были прерваны до первой валидации, лог `furniture_train.log`
неоднозначно соответствует конкретному run-каталогу (ultralytics
переиспользует базовое имя с суффиксом `-2`/`-3` при повторных запусках).

### Наблюдение: добавление CubiCasa5K резко просадило mAP (0.99 → 0.52)

**Это НЕ обязательно баг** — важное уточнение: `_merged` запущен (31.07,
22:53) уже ПОСЛЕ того, как calibrated-парсер CubiCasa5K был готов (31.07,
17:47), то есть, вероятно, использовалась уже исправленная версия
координат, а не баговая v1. Более вероятное объяснение просадки:
- `_merged`/`_merged2` в сумме прошли только **7 эпох с нуля** (3+4), тогда
  как `-3`→`_cont`→`_cont2` в сумме дошли до **7 эпох, но с "тёплым"
  продолжением** от уже сошедшегося на SFPI чекпоинта — плюс на
  SFPI-only датасете модели было ПРОЩЕ (один консистентный источник
  разметки/стиля отрисовки).
- CubiCasa5K добавляет новый визуальный стиль (другая толщина линий,
  другие иконки мебели) — модели нужно время, чтобы обобщиться на оба
  стиля одновременно, 3-4 эпохи явно недостаточно для этого с нуля.
- **Не проверено и требует отдельного вывода**: не повторили без
  CubiCasa5K с тем же малым числом эпох (3-4) для честного сравнения —
  возможно, часть просадки просто из-за "слишком рано мерить".

### Последний запуск упал с OOM

`furniture_train_merged2.log` обрывается на:
```
cv2.error: OpenCV(4.10.0) ... Failed to allocate 58088850 bytes in
function 'cv::OutOfMemoryError'
```
— при декодировании (`cv2.imdecode`) одного из изображений датасета,
после 4 успешных эпох. Похоже на нехватку RAM (не GPU VRAM — ошибка из
`cv::OutOfMemoryError`, общая память процесса), возможно на конкретном
"тяжёлом" файле. Не диагностировано дальше — при возобновлении работы
сначала стоит найти проблемный файл (например, обработкой по одному
с логированием пути перед каждым `imread`) прежде чем перезапускать from
`_merged2/weights/last.pt`.

## Где что лежит

| Что | Путь | В git? |
|---|---|---|
| Таксономия классов | `furniture/configs/furniture_classes.yaml` | да |
| Парсинг источников | `furniture/data_prep/*.py` | да |
| Собранный YOLO-датасет | `furniture/data/furniture_yolo_ds/` | конфиг `data.yaml` — да, картинки — нет (гитигнор на данные) |
| Сырые источники | `furniture/raw/{cubicasa5k,floorplancad,sfpi}/` | нет |
| Чекпоинты (`best.pt`/`last.pt`) | `runs/segment/furniture/output/<run>/weights/` | **нет** (`runs/` в `.gitignore`) |
| Логи обучения | `furniture_train*.log` (корень проекта) | нет |
| Этот журнал | `docs/furniture_experiments_log.md` | да |

## Furniture-модель на UGC — визуально сломана (2026-08-02)

Есть готовый прогон (`runs/segment/furniture/output/ugc_predict/`, 33
файла — все UGC test картинки, инференс через `furniture_yolo11n_cont2`,
судя по всему). Визуальная проверка показала **тяжёлый domain-gap
коллапс**: модель почти на каждой картинке заливает весь план классом
`sink` (raковина) с score 0.4-0.9 — включая подписи площади, номера
комнат, дверные проёмы, даже водяной знак Avito. Ни одного реального
sink на этих местах нет. Похожий паттерн, что и у недообученного
RF-DETR base на ResPlan/CubiCasa val (см. `experiments_log.md`) — модель,
обученная на чистой синтетике (SFPI/FloorPlanCAD/CubiCasa символы мебели),
не переносится на шумные реальные фото вообще. **Количественно на UGC
не оценивалась** (нет GT для мебели на UGC test), только визуально —
но по внешнему виду использовать её как есть нельзя.

### Для сравнения: та же модель на СВОЁМ домене (SFPI val) — работает отлично

Чтобы убедиться, что дело именно в domain gap, а не в фундаментально
сломанной модели, прогнали тот же чекпоинт (`furniture_yolo11n_cont2`,
val mAP50(mask)=0.992) на 10 картинках из его СОБСТВЕННОГО held-out
val-сплита (`furniture/data/furniture_yolo_ds/images/val/`, только
SFPI/FloorPlanCAD — та же доменная выборка, на которой чекпоинт
валидировался при обучении). Результат подтверждает цифры: **точные
маски, confidence 0.84-0.99 почти без исключений**, верно различает
sink/bathtub/sofa/table/bed/chair даже на самых плотных многоквартирных
планах (до 42 объектов на одной картинке). Т.е. модель как таковая
обучена нормально — проблема именно в переносе на реальные фото (шум,
ракурс, освещение), которых не было ни в одном из трёх обучающих
источников.

Визуализации (10 шт.): `docs/report_assets/furniture_sfpi_val_predict/`
(`floor_image_1040/1112/1009/823/1045/430/1244/1015/1241/1120.jpg`).

## OCR площадей на UGC — сравнение движков (2026-08-02)

Три раза (`score_area_{ocr,paddleocr,tesseract}.py`) сверили распознанные
числа с РУЧНОЙ разметкой площади (`furniture/raw/manual_area_gt_template.csv`,
75 строк/67 валидных чисел, по всем видимым на плане комнатам, включая
исключённые категории room/coridor/enterance). Recall — просто "попало ли
истинное число в множество распознанных на всей картинке" (не привязано к
маске комнаты).

| Метод | Recall | Комментарий |
|---|---:|---|
| EasyOCR baseline (`score_area_ocr.py`) | 9/67 = **13.4%** | видит числа, но много промахов; кириллица почти нечитаема (см. `ocr_vis/`) |
| EasyOCR + CLAHE + mag=2-4 | **20.9%** | предобработка (контраст + апскейл) заметно помогает EasyOCR |
| Tesseract digits-only, psm=11 (`score_area_tesseract.py`) | 13/67 = **19.4%** | чуть лучше EasyOCR baseline, всё ещё низко |
| PaddleOCR + CLAHE | **14.9%** | предобработка НЕОЖИДАННО хуже, чем без неё — CLAHE вредит PaddleOCR |
| **PaddleOCR (без обработки)** (`score_area_paddleocr.py`) | 26/67 = **38.8%** | **безоговорочный лидер** — почти вдвое лучше следующего результата |

По категориям (PaddleOCR без обработки, лучший результат): bathroom 4/7
(57%), storage 2/2 (100%, но n=2), balcony 3/8 (38%), room 12/34 (35%),
kitchen 1/4 (25%), restroom 0/2 (0%). Даже у лучшего варианта recall
нестабилен и зависит от категории — на большинстве картинок находится в
лучшем случае половина подписей площади.

Важный вывод про предобработку: CLAHE/апскейл **помогает EasyOCR**
(13.4%→20.9%), но **вредит PaddleOCR** (38.8%→14.9%) — у движков разные
внутренние допущения о входном изображении, единого препроцессинга "для
OCR вообще" не существует, нужно подбирать отдельно под конкретный движок.
Итоговая рекомендация: **PaddleOCR без какой-либо предобработки**.

## Открытые вопросы / что не сделано

- Реранкер как таковой (логика "если детектирован диван — понизь вероятность
  ванной") **не реализован** — есть только детектор мебели, связки с
  основным room-классификатором проекта пока нет.
- CubiCasa5K-калибровка использует внешний код из `avito-toilet/`
  (`cubicasa_calib.py`, `cubicasa_utils_v2.py`) — не наш, не редактируется,
  но и не задокументирован в основном проекте (эти файлы вне
  `claude_instseg_compare/`).
- Не выяснено, почему `furniture_yolo11n`/`-2` не оставили `results.csv`.
- Обучение с CubiCasa5K не доведено до сходимости и прервано крашем —
  нет финальной оценки, помогает ли CubiCasa5K вообще (после нормального
  числа эпох) или её стоит исключить.
