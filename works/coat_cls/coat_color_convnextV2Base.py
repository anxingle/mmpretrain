_base_ = [
    '../../configs/_base_/models/convnext_v2/base.py',
    # '../../configs/_base_/datasets/imagenet_bs64_swin_384.py',
    '../../configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    # '../../configs/_base_/default_runtime.py',
]

# 模型设置 - 二分类任务
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='EfficientNetV2', arch='s'),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=3,  # 灰(gray)/白(white)/黄(yellow) 
#         in_channels=1280,
#         loss=dict(
#             type='CrossEntropyLoss',
#             # 类别权重：处理类别不平衡
#             # ('gray', 'white', 'yellow')数量为(22, 665, 1235)
#             class_weight=[9.1, 2.0, 1.01],  # [gray, white, yellow]
#             loss_weight=1.0),
#         topk=(1,),  # 二分类只看top1准确率
#     ))
# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvNeXt', arch='base', drop_path_rate=0.4, layer_scale_init_value=0., use_grn=True),
    head=dict(
        type='LinearClsHead',
        num_classes=3,  # 灰(gray)/白(white)/黄(yellow) 
        in_channels=1024,
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.2),
        loss=dict(
            type='CrossEntropyLoss',
            # 类别权重：处理类别不平衡
            # ('gray', 'white', 'yellow')数量为(22, 665, 1235)
            class_weight=[9.1, 2.0, 1.01],  # [gray, white, yellow]
            loss_weight=1.0),
        init_cfg=None,
        _delete_ = True # 这将完全删除基础配置中的 head 设置
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.8),
    #     # dict(type='CutMix', alpha=1.0),
    # ]),
)

# 数据预处理设置
data_preprocessor = dict(
    num_classes=3,  # 
    # RGB格式归一化参数（ImageNet标准）
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # 将图像从BGR转换为RGB
    to_rgb=True,
)

# 训练数据增强管道
# 注意：根据舌诊图像特点设计
train_pipeline = [
    dict(type='LoadImageFromFile'),
    
    # 1. 水平翻转（舌体左右对称，可以翻转）
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 随机小角度旋转
    dict(
        type='Rotate',
        angle=12,  # 最大旋转角度12度
        prob=0.6,  # 60%的概率应用旋转
        random_negative_prob=0.5,  # 随机方向旋转正或负12度
        pad_val=0,  # 旋转后空白区域填充黑色，与Pad一致
        interpolation='bilinear'
    ),
    # 随机平移
    dict(type='Translate', magnitude=0.05, prob=0.3, direction='horizontal', pad_val=0),
    dict(type='Translate', magnitude=0.05, prob=0.3, direction='vertical', pad_val=0),
    # 随机缩放、裁剪
    dict(type='RandomResizedCrop', scale=384, crop_ratio_range=(0.95, 1.0), backend='pillow', interpolation='bicubic'),
    # Mixup - 暂时注释掉，避免错误
    # dict(type='Mixup', alpha=0.8, num_classes=3, probs=0.2),
    # XXX: 舌苔颜色 (灰、白、黄) 必须考虑 色彩、亮度、对比度、饱和度 的增强！ 
    # XXX: 慎用 ColorJitter， 会改变颜色！所以必须
    dict(
        type='ColorJitter',
        brightness=0.04, # 亮度扰动范围 98%~1020% 之间随机调整
        contrast=0.04, # 对比度
        saturation=0.04, # 饱和度
        hue=[0.001, 0.01], # 色调偏移范围
        backend='pillow',
    ),
    # 轻微噪声 - 使用 Albumentations 高斯噪声
    dict(
        type='Albu',
        transforms=[
            dict(type='GaussNoise', std_range=(0.02, 0.03), p=0.9),   # 高斯噪声（2%-3%的标准差）
            # 可选：ISO 噪声（相机更像），二选一不要都开
            # dict(type='ISONoise', color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.0),
        ],
        keymap={'img': 'image'}      # 把 mm 的 'img' 映射到 Albu 的 'image'
    ),
    # 2. 轻度高斯模糊（模拟拍摄时的轻微失焦）
    dict(
        type='GaussianBlur',
        magnitude_range=(0.2, 0.5),  # 较轻的模糊程度
        magnitude_std='inf',
        prob=0.8  # 80%概率应用模糊
    ),
    
    # 3. 调整图像尺寸（保持长宽比，避免变形）
    dict(
        type='ResizeEdge',
        scale=384,  # 目标边缘长度
        edge='long',  # 长边缩放到384
        interpolation='lanczos',  # 高质量插值
        backend='cv2'
    ),
    
    # 4. 填充到正方形（避免裁切）
    dict(
        type='Pad',
        size=(384, 384),
        pad_val=0,
        padding_mode='constant'
    ),
    
    dict(type='PackInputs'),
]

# 测试/验证数据管道（无增强）
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='long', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

# 评估指标
val_evaluator = [
    dict(type='Accuracy', topk=(1,)),  # Top-1准确率
    dict(
        type='SingleLabelMetric',
        items=['precision', 'recall', 'f1-score'],
        average=None,  # None表示显示每个类别的指标
    )
]
test_evaluator = val_evaluator

# 数据加载器配置
train_dataloader = dict(
    batch_size=16,  # 根据显存调整
    num_workers=8,
    # metainfo=dict(classes=['gray', 'white', 'yellow]),
    dataset=dict(
        type="CustomDataset",
        data_root='/data-ssd/coating6000_color_v2/',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    # metainfo=dict(classes=['gray', 'white', 'yellow]),
    dataset=dict(
        type="CustomDataset",
        data_root='/data-ssd/coating6000_color_v2_test/',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    # metainfo=dict(classes=['gray', 'white', 'yellow]),
    dataset=dict(
        type="CustomDataset",
        data_root='/data-ssd/coating6000_color_v2_test/',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# schedule setting
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type='AdamW',
        # lr=0.001,  # 这个学习率是正常的
        # lr=2.5e-3,  # 这个学习率是 base.py 中的
        lr=0.001,
        weight_decay=0.02,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

# learning policy - 重写以避免继承问题
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=10,  # ← 缩短预热期
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=500,
        eta_min=0.00001,
        by_epoch=True,
        begin=10,
        end=500,
    )
]

# 训练配置 - 重写
train_cfg = dict(
    by_epoch=True, 
    max_epochs=500,
    val_interval=4
)

# ⚠️ 关键修复：禁用自动学习率缩放
auto_scale_lr = dict(enable=False)  # ← 添加这行！

# 运行时配置
default_scope = 'mmpretrain'

# 默认钩子配置
default_hooks = dict(
    # 记录每次迭代的时间
    timer=dict(type='IterTimerHook'),
    # 每50次迭代打印日志
    logger=dict(type='LoggerHook', interval=50),
    # 启用参数调度器
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 保存检查点
    checkpoint=dict(
        type='CheckpointHook', 
        interval=2,  # 每2个epoch保存
        max_keep_ckpts=40,  # 最多保留10个检查点
        save_best="auto"  # 自动保存最佳模型
    ),
    # 分布式采样器种子
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # 可视化钩子
    visualization=dict(type='VisualizationHook', enable=True),
)

# 自定义钩子：用于可视化训练过程中的图像
custom_hooks = [
    dict(
        type='VisualizationHook',
        enable=True,
        interval=20,  # 每20次迭代可视化一次
        show=False,
        draw_gt=True,
        draw_pred=True,
        out_dir='/data-ssd/logs/vis_teeth_normal',  # 可视化输出目录
    ),
]

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 可视化后端
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# 日志级别
log_level = 'INFO'

# 加载预训练模型（EfficientNetV2-XL在ImageNet上的预训练权重）
load_from = "/home/an/mmpretrain/works/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth"
# 不恢复训练（从头开始）
resume = False

# 随机性配置
randomness = dict(seed=42, deterministic=False)  # 设置种子以保证可重复性

# 工作目录（保存日志和检查点）
work_dir = '/data-ssd/logs/coat_color_convNextV2_base'
