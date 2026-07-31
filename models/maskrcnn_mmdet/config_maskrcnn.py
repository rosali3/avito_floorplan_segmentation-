# Mask R-CNN R50-FPN, дообучение (transfer learning) с COCO-претрейна на наш
# 7-классовый instance segmentation датасет (combined_out -> train_coco.json/valid_coco.json).
#
# ВАЖНО про переносимость (напр. запуск на GPU-сервере позже): этот файл НЕ
# содержит абсолютных путей и не хардкодит список классов "на постоянку" —
# все датасет-специфичные поля (data_root, ann_file, work_dir, metainfo.classes,
# num_classes, load_from) подставляются программно в train.py ПОСЛЕ
# Config.fromfile(), из ЕДИНОГО источника правды configs/paths.yaml /
# configs/classes.yaml (через data_prep/coco_utils.load_paths/load_classes).
# Если поменяются пути или таксономия — правь только paths.yaml/classes.yaml,
# этот файл трогать не нужно. Значения ниже — просто заглушки-дефолты на
# случай, если кто-то запустит mmdet напрямую на этом конфиге, минуя train.py.

_base_ = "./base_config/mask-rcnn_r50_fpn_1x_coco.py"

_PLACEHOLDER_CLASS_NAMES = ("living", "bedroom", "bathroom", "kitchen", "balcony", "wall", "opening")
_PLACEHOLDER_NUM_CLASSES = len(_PLACEHOLDER_CLASS_NAMES)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=_PLACEHOLDER_NUM_CLASSES),
        mask_head=dict(num_classes=_PLACEHOLDER_NUM_CLASSES),
    )
)

metainfo = dict(classes=_PLACEHOLDER_CLASS_NAMES)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(metainfo=metainfo, data_prefix=dict(img="")),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(metainfo=metainfo, data_prefix=dict(img="")),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="CocoMetric", metric=["bbox", "segm"])
test_evaluator = val_evaluator

# Классический баттлан для transfer learning на небольшом числе классов:
# 24 эпохи, step-decay на 16/22. Поменяй под свои сроки через --epochs в train.py
# (train.py переопределяет max_epochs/milestones программно, тут — дефолт).
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=1)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1),
]
optim_wrapper = dict(optimizer=dict(lr=0.0025))  # ниже стандартного 0.02 (8x меньше batch, transfer learning)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3,
                     save_best="coco/segm_mAP", rule="greater"),
    logger=dict(type="LoggerHook", interval=20),
)

visualizer = dict(
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(type="TensorboardVisBackend"),
        dict(type="MLflowVisBackend", exp_name="maskrcnn"),  # train.py переопределяет save_dir
    ]
)
