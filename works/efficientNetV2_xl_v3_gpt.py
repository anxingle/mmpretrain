# Configuration for multi-label tongue morphology classification
# using EfficientNetV2-XL

_base_ = [
    '../configs/_base_/models/efficientnet_v2/efficientnetv2_xl.py',
    '../configs/_base_/schedules/imagenet_bs256.py',
    '../configs/_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNetV2', arch='xl'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        # Multi-label head with sigmoid based classification
        type='MultiLabelLinearClsHead',
        num_classes=4,  # teeth-mark / dot-prick / crack / normal
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
        # positive predictions are determined by threshold=0.5 by default
    ),
)

# # dataset settings
# data_preprocessor = dict(
#     num_classes=4,
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     to_rgb=True,
#     # to_onehot=True,  # convert label lists to one-hot vectors for multi-label
#     # Note: ClsDataPreprocessor doesn't support multi_label parameter
#     # Multi-label handling is done in the loss function and head
# )
data_preprocessor = dict(
    type='ClsDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# augmentation policy for tongue morphology
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # symmetry augmentation
    dict(
        type='GaussianBlur',
        magnitude_range=(0.4, 0.7),
        magnitude_std='inf',
        prob=0.6,
    ),
    dict(
        type='ResizeEdge',
        scale=384,
        edge='long',
        interpolation='lanczos',
        backend='cv2',
    ),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

custom_hooks = [
    dict(
        type='VisualizationHook',
        enable=True,
        interval=10,
        show=False,
        draw_gt=True,
        draw_pred=True,
        out_dir='./tmp/512_shape_multilabel',
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='long', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

# multi-label evaluation metrics
# val_evaluator = dict(type='MultiLabelMetric', average='macro')
val_evaluator = [
    dict(type='MultiLabelMetric', average='macro'),  # 宏平均
    dict(type='MultiLabelMetric', average='micro'),  # 微平均
    dict(type='AveragePrecision', average='macro'),  # mAP
]
test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='MultiLabelDataset',
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_train_fixed',
        ann_file='train_annotations.json',  # OpenMMLab 2.0 style annotation
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
    batch_size=32,
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

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    accumulative_counts=2,  # 每 2 个 iteration 累积一次梯度再更新: 解决显存不足问题，模拟更大的 batch_size; 保持训练稳定；
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=800,
        eta_min=1e-5,
        by_epoch=True,
        begin=30,
        end=800,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=10)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = None

# default runtime
default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=5, max_keep_ckpts=10,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# environment settings
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

log_level = 'INFO'

# load ImageNet pretrained weights
load_from = ('/home/an/mmpretrain/works/'
             'efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth')
resume = False
randomness = dict(seed=None, deterministic=False)
