#!/usr/bin/env bash
# Конвертер данных — весь шаг 1 из README.md одной командой. Работает и
# локально (Windows/Git Bash), и на Linux-сервере позже (пути берутся из
# configs/paths.yaml или переменных окружения CLAUDE_COMBINED_OUT_ROOT /
# CLAUDE_UGC_LABELED_ROOT / CLAUDE_PROJECT_ROOT — см. README "Перенос на сервер").
#
# Зависимости (лёгкие, без тяжёлых ML-фреймворков):
#   pip install -r requirements-common.txt
#
# Запуск (из корня claude_instseg_compare/):
#   bash run_data_prep.sh
#   # только train/valid без ugc/yolo/rfdetr (напр. для sanity-проверки):
#   bash run_data_prep.sh --skip-ugc --skip-yolo --skip-rfdetr
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SKIP_UGC=0
SKIP_YOLO=0
SKIP_RFDETR=0
BUILD_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --skip-ugc) SKIP_UGC=1 ;;
    --skip-yolo) SKIP_YOLO=1 ;;
    --skip-rfdetr) SKIP_RFDETR=1 ;;
    --limit=*) BUILD_ARGS+=(--limit "${arg#*=}") ;;
    *) echo "неизвестный флаг: $arg" >&2; exit 1 ;;
  esac
done

echo "=== [1/4] train_coco.json / valid_coco.json (instances из combined_out) ==="
python data_prep/build_train_val_coco.py "${BUILD_ARGS[@]}"

if [ "$SKIP_UGC" -eq 0 ]; then
  echo "=== [2/4] test_coco.json (ugc_labeled, объединённый test) ==="
  python data_prep/prepare_ugc_test.py
else
  echo "=== [2/4] пропущено (--skip-ugc) ==="
fi

if [ "$SKIP_YOLO" -eq 0 ]; then
  echo "=== [3/4] YOLO-seg формат ==="
  python data_prep/coco_to_yolo_seg.py
else
  echo "=== [3/4] пропущено (--skip-yolo) ==="
fi

if [ "$SKIP_RFDETR" -eq 0 ]; then
  echo "=== [4/4] rfdetr dataset_dir ==="
  python data_prep/prepare_rfdetr_dataset.py
else
  echo "=== [4/4] пропущено (--skip-rfdetr) ==="
fi

echo "=== готово. Дальше — обучение конкретной модели, см. README.md раздел 2 ==="
