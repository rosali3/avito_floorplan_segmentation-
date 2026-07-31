# Конвертер данных — весь шаг 1 из README.md одной командой (нативный Windows,
# без Git Bash). Линуксовый эквивалент для сервера — run_data_prep.sh.
#
# Зависимости: pip install -r requirements-common.txt
#
# Запуск (из корня claude_instseg_compare/):
#   powershell -ExecutionPolicy Bypass -File run_data_prep.ps1
#   powershell -ExecutionPolicy Bypass -File run_data_prep.ps1 -SkipUgc -SkipYolo -SkipRfdetr

param(
    [switch]$SkipUgc,
    [switch]$SkipYolo,
    [switch]$SkipRfdetr,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== [1/4] train_coco.json / valid_coco.json (instances из combined_out) ==="
if ($Limit -gt 0) {
    python data_prep/build_train_val_coco.py --limit $Limit
} else {
    python data_prep/build_train_val_coco.py
}

if (-not $SkipUgc) {
    Write-Host "=== [2/4] test_coco.json (ugc_labeled, объединённый test) ==="
    python data_prep/prepare_ugc_test.py
} else {
    Write-Host "=== [2/4] пропущено (-SkipUgc) ==="
}

if (-not $SkipYolo) {
    Write-Host "=== [3/4] YOLO-seg формат ==="
    python data_prep/coco_to_yolo_seg.py
} else {
    Write-Host "=== [3/4] пропущено (-SkipYolo) ==="
}

if (-not $SkipRfdetr) {
    Write-Host "=== [4/4] rfdetr dataset_dir ==="
    python data_prep/prepare_rfdetr_dataset.py
} else {
    Write-Host "=== [4/4] пропущено (-SkipRfdetr) ==="
}

Write-Host "=== готово. Дальше — обучение конкретной модели, см. README.md раздел 2 ==="
