"""
Обучение Mask R-CNN (MMDetection) — классический баттлан для сравнения.

Все датасет-специфичные поля (пути к данным, work_dir, список классов,
load_from) подставляются здесь программно из configs/paths.yaml /
configs/classes.yaml — config_maskrcnn.py сам по себе НЕ содержит абсолютных
путей, поэтому при переносе на другую машину (напр. на GPU-сервер) достаточно
поправить только configs/paths.yaml (или выставить переменные окружения
CLAUDE_COMBINED_OUT_ROOT / CLAUDE_UGC_LABELED_ROOT / CLAUDE_PROJECT_ROOT,
см. data_prep/coco_utils.py) — этот файл трогать не нужно.

Перед запуском см. SETUP.md (отдельный venv, mim install, mim download base config)
и выполни (из корня claude_instseg_compare/):
    python data_prep/build_train_val_coco.py

Запуск:
    python models/maskrcnn_mmdet/train.py --load-from models/maskrcnn_mmdet/base_config/<имя_чекпоинта>.pth
    python models/maskrcnn_mmdet/train.py --epochs 36 --batch-size 4 --load-from ...
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # только вторая GPU (индекс 1, общий сервер) — см. progress.md

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_prep"))
from coco_utils import load_classes, load_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=2,
                     help="под 8GB VRAM держи 2; на карте с большей памятью можно 4-8")
    ap.add_argument("--load-from", default=None,
                     help="COCO-претрейн чекпоинт из 'mim download' (см. SETUP.md); "
                          "если не задан, обучение идёт со случайной инициализации головы "
                          "поверх ImageNet backbone из base_config (заметно медленнее сходится)")
    ap.add_argument("--data-dir", default=None,
                     help="папка с train_coco.json/valid_coco.json — по умолчанию "
                          "paths.yaml:derived.data_dir (укажи data_v2 для исправленных данных)")
    ap.add_argument("--no-mlflow", action="store_true", help="выключить MLflow-логирование")
    args = ap.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner

    paths = load_paths()
    classes_cfg = load_classes()
    fg = classes_cfg["foreground_classes"]
    class_names = tuple(fg[i] for i in sorted(fg, key=int))
    num_classes = len(class_names)
    metainfo = dict(classes=class_names)

    data_root_images = str(Path(paths["combined_out_root"]) / "images") + "/"
    data_dir = Path(args.data_dir) if args.data_dir else Path(paths["derived"]["data_dir"])
    data_json_dir = str(data_dir) + "/"
    print(f"[maskrcnn train] data_dir = {data_dir}")
    work_dir = str(Path(paths["derived"]["output_dir"]) / "maskrcnn_mmdet" / "checkpoints")

    cfg_path = Path(__file__).parent / "config_maskrcnn.py"
    cfg = Config.fromfile(str(cfg_path))

    cfg.model.roi_head.bbox_head.num_classes = num_classes
    cfg.model.roi_head.mask_head.num_classes = num_classes

    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.dataset.metainfo = metainfo
    cfg.train_dataloader.dataset.data_root = data_root_images
    cfg.train_dataloader.dataset.ann_file = data_json_dir + "train_coco.json"

    cfg.val_dataloader.dataset.metainfo = metainfo
    cfg.val_dataloader.dataset.data_root = data_root_images
    cfg.val_dataloader.dataset.ann_file = data_json_dir + "valid_coco.json"
    cfg.test_dataloader = cfg.val_dataloader

    cfg.val_evaluator.ann_file = data_json_dir + "valid_coco.json"
    cfg.test_evaluator = cfg.val_evaluator

    cfg.work_dir = work_dir

    if args.no_mlflow:
        cfg.visualizer.vis_backends = [b for b in cfg.visualizer.vis_backends
                                        if b.get("type") != "MLflowVisBackend"]
    else:
        for b in cfg.visualizer.vis_backends:
            if b.get("type") == "MLflowVisBackend":
                b["save_dir"] = str(Path(work_dir) / "mlruns")
        print(f"[maskrcnn train] MLflow: file:{Path(work_dir) / 'mlruns'}")

    if args.load_from:
        cfg.load_from = args.load_from
    else:
        print("[maskrcnn train] ВНИМАНИЕ: --load-from не задан — без COCO-претрейна "
              "голова/маска-декодер учатся с нуля, дольше сходятся. См. SETUP.md.")

    cfg.train_cfg.max_epochs = args.epochs
    cfg.param_scheduler[1].end = args.epochs
    cfg.param_scheduler[1].milestones = [int(args.epochs * 0.67), int(args.epochs * 0.92)]

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()

    work_dir_p = Path(cfg.work_dir)
    # mmengine сохраняет best_coco_segm_mAP_epoch_N.pth и epoch_<last>.pth
    best_ckpts = sorted(work_dir_p.glob("best_coco_segm_mAP_epoch_*.pth"))
    epoch_ckpts = sorted(work_dir_p.glob("epoch_*.pth"), key=lambda p: int(p.stem.split("_")[-1]))
    if best_ckpts:
        shutil.copyfile(best_ckpts[-1], work_dir_p / "best.pth")
    else:
        print(f"[maskrcnn train] ВНИМАНИЕ: не нашёл best_coco_segm_mAP_epoch_*.pth в {work_dir_p}")
    if epoch_ckpts:
        shutil.copyfile(epoch_ckpts[-1], work_dir_p / "final.pth")
    else:
        print(f"[maskrcnn train] ВНИМАНИЕ: не нашёл epoch_*.pth в {work_dir_p}")

    print(f"[maskrcnn train] done. логи (json/tensorboard) -> {work_dir_p}")
    print(f"[maskrcnn train] best -> {work_dir_p / 'best.pth'}, final -> {work_dir_p / 'final.pth'}")


if __name__ == "__main__":
    main()
