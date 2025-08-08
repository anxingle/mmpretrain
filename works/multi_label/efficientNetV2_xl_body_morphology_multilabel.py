# 舌体形态多标签分类配置文件
# 任务：齿痕舌(teeth)、点刺舌(swift)、裂纹舌(crack)、正常(normal) 四类多标签分类
# 模型：EfficientNetV2-XL
# 说明：一个舌体可能同时具有多种形态特征，也可能完全正常

_base_ = [
    '../configs/_base_/models/efficientnet_v2/efficientnetv2_s.py',
    '../configs/_base_/schedules/imagenet_bs256.py',
    '../configs/_base_/default_runtime.py',
]

# ============================================================================
# 模型配置 - 多标签分类
# ============================================================================
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNetV2',
        arch='xl',
        drop_path_rate=0.2,  # 增加正则化，防止过拟合
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth',
            prefix='backbone'
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',  # 多标签分类头MultiLabelLinearHead
        num_classes=4,  # 齿痕舌、点刺舌、裂纹舌、正常
        in_channels=1280,
        loss=dict(
            type='AsymmetricLoss',  # 非对称损失，适合多标签不平衡数据
            gamma_neg=4,  # 负样本的focal参数
            gamma_pos=1,  # 正样本的focal参数
            clip=0.05,    # 梯度裁剪
            eps=1e-8,
            disable_torch_grad_focal_loss=True
        ),
        # 备选损失函数：BCEWithLogitsLoss
        # loss=dict(
        #     type='BCEWithLogitsLoss',
        #     use_sigmoid=True,
        #     reduction='mean',
        #     class_weight=[1.0, 1.0, 1.0, 1.0],  # 可根据数据分布调整
        #     pos_weight=None  # 可设置正样本权重
        # ),
    )
)

# ============================================================================
# 数据预处理配置
# ============================================================================
data_preprocessor = dict(
    num_classes=4,  # 舌体形态：齿痕/点刺/裂纹/正常
    mean=[123.675, 116.28, 103.53],  # ImageNet标准化参数
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# ============================================================================
# 数据增强配置 - 针对舌体形态特征优化
# ============================================================================
# 舌体形态分类的特殊要求：
# 1. 保持舌体完整性，禁用RandomResizedCrop
# 2. 保持边缘和纹理特征清晰，适度使用几何变换
# 3. 齿痕、点刺、裂纹等特征对形状和纹理敏感
# 4. 可以适当增强对比度来突出纹理细节

train_pipeline = [
    dict(type='LoadImageFromFile'),

    # 几何变换 - 适度使用，保持形态特征
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotation', angle=(-15, 15), prob=0.3),  # 小角度旋转

    # 颜色增强 - 轻微调整，保持纹理可见性
    dict(
        type='ColorJitter',
        brightness=0.15,   # 适度亮度调整，突出纹理
        contrast=0.15,     # 适度对比度调整，增强边缘
        saturation=0.10,   # 轻微饱和度调整
        hue=0.02,          # 很小的色调调整
        backend='pillow',
        prob=0.4
    ),

    # 轻微模糊 - 保持边缘特征
    dict(
        type='GaussianBlur',
        magnitude_range=(0.3, 0.6),
        magnitude_std='inf',
        prob=0.3  # 降低模糊概率，保持纹理清晰
    ),

    # 尺寸调整 - 保持长宽比
    dict(
        type='ResizeEdge',
        scale=384,           # 目标边缘长度
        edge='long',         # 长边缩放
        interpolation='lanczos',  # 高质量插值
        backend='cv2'
    ),
    dict(
        type='Pad',
        size=(384, 384),
        pad_val=0,
        padding_mode='constant'
    ),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='long', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

# ============================================================================
# 数据加载器配置
# ============================================================================
train_dataloader = dict(
    batch_size=6,  # 降低batch size，XL模型显存需求大
    num_workers=8,
    dataset=dict(
        type="MultiLabelDataset",  # 改为多标签数据集
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_train',
        ann_file='train_annotations.json',  # 指定训练集标注文件
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="MultiLabelDataset",  # 改为多标签数据集
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_train',
        ann_file='val_annotations.json',  # 指定验证集标注文件
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type="MultiLabelDataset",  # 改为多标签数据集
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_test',
        ann_file='test_annotations.json',  # 指定测试集标注文件
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# ============================================================================
# 评估器配置 - 多标签评估指标
# ============================================================================
val_evaluator = [
    dict(type='MultiLabelMetric', average='macro'),  # 宏平均
    dict(type='MultiLabelMetric', average='micro'),  # 微平均
    dict(type='AveragePrecision', average='macro'),  # mAP
]
test_evaluator = val_evaluator

# ============================================================================
# 优化器配置
# ============================================================================
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0005,  # 降低学习率，多标签任务更敏感
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0
    ),
)

# ============================================================================
# 学习率调度
# ============================================================================
param_scheduler = [
    # 线性预热
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=20,  # 预热20个epoch
        convert_to_iter_based=True,
    ),
    # 余弦退火
    dict(
        type="CosineAnnealingLR",
        T_max=200,  # 总训练轮数
        eta_min=0.00001,
        by_epoch=True,
        begin=20,
        end=200,
    )
]

# ============================================================================
# 训练配置
# ============================================================================
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# ============================================================================
# 钩子配置
# ============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=5,
        save_best="auto",
        rule='greater'  # 多标签任务通常希望指标越大越好
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# 自定义钩子 - 可视化多标签预测结果
custom_hooks = [
    dict(
        type='VisualizationHook',
        enable=True,
        interval=20,
        show=False,
        draw_gt=True,
        draw_pred=True,
        out_dir='./tmp/xl_body_morphology_multilabel',
    ),
]

# ============================================================================
# 环境配置
# ============================================================================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ============================================================================
# 可视化配置
# ============================================================================
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# ============================================================================
# 日志和其他配置
# ============================================================================
log_level = 'INFO'
log_processor = dict(by_epoch=True)

# 预训练权重
load_from = "/home/an/mmpretrain/works/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth"
resume = False

# 随机种子
randomness = dict(seed=42, deterministic=False)

# ============================================================================
# 数据集元信息
# ============================================================================
dataset_meta = dict(
    classes=['齿痕舌', '点刺舌', '裂纹舌', '正常'],
    # 多标签分类说明：
    # - 齿痕舌(teeth): 舌边缘有明显的齿印痕迹，呈锯齿状
    # - 点刺舌(swift): 舌面有红色点状突起，质地较硬
    # - 裂纹舌(crack): 舌面有深浅不一的裂纹或沟纹
    # - 正常(normal): 无明显形态异常
    #
    # 标签格式示例：
    # [1, 0, 1, 0] - 同时有齿痕和裂纹
    # [0, 1, 0, 0] - 只有点刺
    # [0, 0, 0, 1] - 正常舌体
    # [1, 1, 1, 0] - 同时有齿痕、点刺、裂纹
    paper_info=dict(
        title='Multi-label Tongue Body Morphology Classification for TCM',
        authors='TCM Research Team',
        year='2024'
    )
)

# ============================================================================
# 多标签分类特殊说明
# ============================================================================
# 1. 数据标注要求：
#    - 每个图像需要多标签标注，格式为 [teeth, swift, crack, normal]
#    - 如果舌体正常，则标注为 [0, 0, 0, 1]
#    - 如果有多种形态异常，对应位置标注为1，normal标注为0
#
# 2. 损失函数选择：
#    - AsymmetricLoss: 专为多标签不平衡数据设计
#    - BCEWithLogitsLoss: 经典多标签损失函数
#    - 可根据数据分布调整class_weight和pos_weight
#
# 3. 评估指标：
#    - mAP (mean Average Precision): 多标签分类的主要指标
#    - Macro/Micro F1: 宏平均和微平均F1分数
#    - 每个类别的精确率、召回率、F1分数
#
# 4. 训练技巧：
#    - 使用较小的学习率，多标签任务对超参数更敏感
#    - 增加正则化，防止过拟合
#    - 保持图像完整性，避免破坏形态特征
#    - 适度数据增强，保持纹理和边缘清晰