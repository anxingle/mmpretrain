# Multi-label Tongue Morphology Classification (EfficientNetV2-XL)
# Classes: ['teeth', 'swift', 'crack', 'normal']

_base_ = [
    '../../configs/_base_/models/efficientnet_v2/efficientnetv2_xl.py',
    '../../configs/_base_/default_runtime.py',
]

default_scope = 'mmpretrain'

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNetV2',
        arch='xl',
        drop_path_rate=0.2,
        init_cfg=dict(
            type='Pretrained',
            # 只保留一处加载（本地文件），避免二次加载冲突
            checkpoint='/home/an/mmpretrain/works/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth',
            prefix='backbone',
        ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=4,
        in_channels=1280,
        thr=0.4,  # 与 evaluator 保持一致
        loss=dict(
            type='AsymmetricLoss',
            use_sigmoid=True,
            gamma_pos=0.0,
            gamma_neg=4.0,
            clip=0.05,
            eps=1e-8,
        ),
        # 如需先验稳妥，可改为：
        # loss=dict(type='BCEWithLogitsLoss', reduction='mean'),
    ),
)

# -----------------------------------------------------------------------------
# Data preprocessor
# -----------------------------------------------------------------------------
data_preprocessor = dict(
    type='ClsDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# -----------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='ColorJitter',
        brightness=0.28,
        contrast=0.28,   # 比 0.38 更稳，减少对裂纹/点刺纹理的破坏
        saturation=0.20,
        hue=[0.00001, 0.0199],
        backend='pillow',
    ),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 0.4),
        magnitude_std='inf',
        prob=0.3,
    ),
    dict(type='ResizeEdge', scale=384, edge='long', interpolation='lanczos', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='long', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

# -----------------------------------------------------------------------------
# Dataloaders
# -----------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='MultiLabelDataset',
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_train_fixed',
        ann_file='train_annotations.json',  # JSONList: {metainfo, data_list:[{img_path, gt_label}]}
        pipeline=train_pipeline,
        metainfo=dict(classes=['teeth', 'swift', 'crack', 'normal']),
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='MultiLabelDataset',
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_train_fixed',
        ann_file='val_annotations.json',
        pipeline=test_pipeline,
        metainfo=dict(classes=['teeth', 'swift', 'crack', 'normal']),
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='MultiLabelDataset',
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_test_fixed',
        ann_file='test_annotations.json',
        pipeline=test_pipeline,
        metainfo=dict(classes=['teeth', 'swift', 'crack', 'normal']),
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# -----------------------------------------------------------------------------
# Evaluators (阈值与 head 对齐)
# -----------------------------------------------------------------------------
val_evaluator = [
    dict(type='MultiLabelMetric', average='macro', thr=0.4),
    dict(type='MultiLabelMetric', average='micro', thr=0.4),
    dict(type='AveragePrecision', average='macro'),
    dict(type='MultiLabelMetric', average=None, thr=0.4),  # per-class
    dict(type='AveragePrecision', average=None),           # per-class AP
]
test_evaluator = val_evaluator

# -----------------------------------------------------------------------------
# Optimizer / Paramwise
# -----------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.01,   # 更稳的衰减
        eps=1e-8,
        betas=(0.9, 0.999),
    ),
    # 用通配符强制禁用 norm/bias 的 decay，兼容不同版本命名
    paramwise_cfg=dict(
        custom_keys={
            '.bn.': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.8),
            'head': dict(lr_mult=1.3),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    # 如需梯度累积（等效更大 batch），你的 mmengine 支持的话可加：
    # accumulative_counts=2,
)

# -----------------------------------------------------------------------------
# Schedulers
# -----------------------------------------------------------------------------
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,                 # 缩短 warmup，加速进入稳定阶段
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=800,             # 与 max_epochs 对齐
        eta_min=2e-5,
        by_epoch=True,
        begin=5,
        end=800,
    ),
]

# -----------------------------------------------------------------------------
# Training / Runtime
# -----------------------------------------------------------------------------
train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=5)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=5,
        save_best='auto',   # 若日志里 mAP 键名稳定，可改成 'mAP'
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

custom_hooks = [
    dict(
        type='VisualizationHook',
        enable=True,
        interval=20,
        show=False,
        draw_gt=True,
        draw_pred=True,
        out_dir='./tmp/tongue_multilabel_vis',
    ),
]

env_cfg = dict(
    cudnn_benchmark=True,   # 固定输入尺寸时建议开启
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
log_processor = dict(by_epoch=True)

# 不再使用 load_from，避免二次加载；只通过 backbone.init_cfg 加载
# load_from = None

resume = False
randomness = dict(seed=42, deterministic=False)

# -----------------------------------------------------------------------------
# Dataset meta (doc purpose)
# -----------------------------------------------------------------------------
dataset_meta = dict(
    classes=['齿痕舌', '点刺舌', '裂纹舌', '正常'],
    paper_info=dict(
        title='Multi-label Tongue Body Morphology Classification for TCM',
        authors='TCM Research Team',
        year='2024',
    ),
)