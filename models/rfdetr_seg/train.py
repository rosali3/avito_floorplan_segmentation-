"""
Обучение RF-DETR-Seg (Medium) через библиотеку `rfdetr` (Roboflow).

Перед запуском:
    pip install rfdetr
    python data_prep/build_train_val_coco.py
    python data_prep/prepare_rfdetr_dataset.py

Запуск (из корня claude_instseg_compare/):
    python models/rfdetr_seg/train.py
    # на менее мощной карте (наша локальная RTX 3080 8GB) при OOM уменьшай
    # batch-size и увеличивай grad-accum-steps, сохраняя произведение = 16:
    python models/rfdetr_seg/train.py --batch-size 2 --grad-accum-steps 8

Метрики по эпохам: rfdetr сам пишет {output_dir}/metrics.csv (CSVLogger по
умолчанию) + TensorBoard-логи при tensorboard=True (смотреть:
tensorboard --logdir output/rfdetr_seg/checkpoints).
Чекпоинты: {output_dir}/checkpoint_best_total.pth (лучший) и
{output_dir}/checkpoint.pth (последняя эпоха) — имена см. в логе первого
запуска и, если библиотека их поменяла в новой версии, поправь FINAL_CKPT_NAME/
BEST_CKPT_NAME ниже.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
from coco_utils import load_paths  # noqa: E402

BEST_CKPT_NAME = "checkpoint_best_total.pth"
FINAL_CKPT_NAME = "checkpoint.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", default=None, help="путь к checkpoint.pth для дообучения")
    ap.add_argument("--no-early-stopping", action="store_true",
                     help="rfdetr по умолчанию early_stopping=False — мы включаем True, "
                          "передай этот флаг чтобы вернуть выключенным")
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--dataset-dir", default=None,
                     help="переопределить dataset_dir — по умолчанию "
                          "paths.yaml:derived.rfdetr_dataset_dir")
    args = ap.parse_args()

    eff_bs = args.batch_size * args.grad_accum_steps
    print(f"[rfdetr train] batch_size={args.batch_size} x grad_accum_steps="
          f"{args.grad_accum_steps} = эффективный batch {eff_bs} "
          f"(рекомендация Roboflow: держать эту сумму ~16)")

    from rfdetr import RFDETRSegMedium  # импорт после парсинга аргументов -- тяжёлая либа

    paths = load_paths()
    dataset_dir = args.dataset_dir if args.dataset_dir else paths["derived"]["rfdetr_dataset_dir"]
    print(f"[rfdetr train] dataset_dir = {dataset_dir}")
    output_dir = Path(paths["derived"]["output_dir"]) / "rfdetr_seg" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRSegMedium()
    train_kwargs = dict(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=str(output_dir),
        tensorboard=True,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)

    # Копируем в предсказуемые имена best.pth/final.pth, чтобы infer_and_eval.py
    # и все остальные models/* были единообразны независимо от версии rfdetr.
    best_src = output_dir / BEST_CKPT_NAME
    final_src = output_dir / FINAL_CKPT_NAME
    if best_src.is_file():
        shutil.copyfile(best_src, output_dir / "best.pth")
    else:
        print(f"[rfdetr train] ВНИМАНИЕ: не нашёл {best_src} — проверь актуальное имя "
              f"лучшего чекпоинта в {output_dir} и поправь BEST_CKPT_NAME в этом файле.")
    if final_src.is_file():
        shutil.copyfile(final_src, output_dir / "final.pth")
    else:
        print(f"[rfdetr train] ВНИМАНИЕ: не нашёл {final_src} — проверь актуальное имя "
              f"финального чекпоинта в {output_dir} и поправь FINAL_CKPT_NAME в этом файле.")

    print(f"[rfdetr train] done. metrics.csv и TensorBoard-логи -> {output_dir}")


if __name__ == "__main__":
    main()
