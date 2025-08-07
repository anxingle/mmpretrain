# tongue_shape_multilabel.py

_base_ = [
    '../configs/_base_/models/efficientnet_v2/efficientnetv2_s.py',
    '../configs/_base_/schedules/imagenet_bs256.py',
    '../configs/_base_/default_runtime.py',
]

# --- 1. 模型设置 (Model Settings) ---
# 核心改动：适配多标签分类任务
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNetV2', arch='xl',
        # init_cfg 会自动处理来自 _base_ 的预训练权重加载。
        # 如果您想强制使用本地权重，请在文件末尾使用 `load_from` 字段。
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        # 类别数仍为 4: teeth, swift, crack, normal (齿痕, 点刺, 裂纹, 正常)
        num_classes=4,
        # EfficientNetV2-XL 的 neck 前输出通道数，保持不变
        in_channels=1280,
        # --- 核心修改：损失函数 ---
        # 对于多标签分类，每个标签的预测是独立的（一张图可以同时是“齿痕舌”和“裂纹舌”）。
        # 因此，我们必须使用 BCEWithLogitsLoss (带 Sigmoid 的二元交叉熵)，而不是 CrossEntropyLoss。
        # mmpretrain 中，最直接的方式就是使用 `BCEWithLogitsLoss`。
        loss=dict(
            type='BCEWithLogitsLoss', # 明确使用 BCE 损失，这是多标签分类的标准选择。
            loss_weight=1.0
        ),
        # `class_weight` 和 `topk` 参数是单标签分类 `CrossEntropyLoss` 的特性，在多标签 BCE 损失下不再适用，因此需要移除。
    ))

# --- 2. 数据预处理器 (Data Preprocessor) ---
data_preprocessor = dict(
    num_classes=4, # 类别数
    # 使用 ImageNet 默认的均值和方差进行归一化，这是使用预训练模型的标准做法
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # 将 OpenCV 读取的 BGR 格式图像转换为 RGB
    to_rgb=True,
)

# --- 3. 训练数据增强管道 (Train Pipeline) ---
# 您原有的数据增强策略非常棒，特别是为保持舌体完整性所做的调整。
# 对于舌体形态分类，形态比颜色更重要，所以 ColorJitter 和 GaussianBlur 是合理的扰动。
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # 50% 的概率进行水平翻转，增加数据多样性
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 轻微的颜色扰动，让模型对光照、色温变化更鲁棒。
    dict(
        type='ColorJitter',
        brightness=0.10,
        contrast=0.09,
        saturation=0.109,
        hue=[0.00001, 0.0099],
        backend='pillow',
    ),
    # 随机高斯模糊，模拟部分图像不清晰的情况
    dict(
        type='GaussianBlur',
        magnitude_range=(0.4, 0.7),
        magnitude_std='inf',
        prob=0.6
    ),
    # 保持纵横比缩放，然后用黑色像素填充至正方形。
    # 这是避免裁切导致舌体信息丢失的绝佳方法。
    dict(
        type='ResizeEdge',
        scale=384,
        edge='long',
        interpolation='lanczos',
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

# --- 4. 测试/验证数据管道 (Test/Validation Pipeline) ---
# 验证和测试时，不做随机的数据增强，只进行尺寸调整和归一化。
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='long', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]


# --- 5. 评估器 (Evaluator) ---
# --- 核心修改：评估指标 ---
# 对于多标签任务，不能再用简单的 `Accuracy`。
# 我们需要使用 `MultiLabelMetric` 来计算精确率(Precision), 召回率(Recall), F1分数(F1-score)等。
val_evaluator = dict(
    type='MultiLabelMetric',
    # `thr` 是一个阈值。当模型对某个类别的预测分数高于该阈值时，我们才认为图片属于该类别。
    # 0.5 是一个常用的初始值，您可以在训练后根据验证集的 PR 曲线进行调整以获得最佳 F1 分数。
    thr=0.5,
    # `items` 指定了需要计算和在日志中显示的指标。
    items=['precision', 'recall', 'f1-score', 'support'],
)
test_evaluator = val_evaluator


# --- 6. 数据加载器 (Dataloaders) ---
# !!! 重要提示: 请将下面的 `data_root` 和 `ann_file` 路径修改为您自己的数据集路径 !!!
#
# **多标签标注文件格式说明:**
# 您需要创建一个 .txt 标注文件，每一行的格式为:
# `图片相对路径 标签索引1 标签索引2 ...` (标签索引之间用空格隔开)
# 假设类别索引为: teeth-0, swift-1, crack-2, normal-3
# 示例 (`train.txt`):
#   class1/image_001.jpg 0
#   class2/image_002.jpg 1 2
#   class3/image_003.jpg 3
#
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="CustomDataset",
        # !!! 修改此路径为您的训练集根目录
        data_root='path/to/your/tongue_shape_dataset/train',
        # !!! 指定您的多标签标注文件
        ann_file='meta/train.txt',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="CustomDataset",
        # !!! 修改此路径为您的验证集根目录
        data_root='path/to/your/tongue_shape_dataset/val',
        # !!! 指定您的多标签标注文件
        ann_file='meta/val.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = val_dataloader # 测试集通常和验证集使用相同的加载器配置

# --- 7. 优化器和学习率调度器 (Optimizer & Scheduler) ---
# 您原有的配置 (SGD + 预热 + 余弦退火) 非常优秀，是经过验证的稳定有效的训练策略，予以保留。
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

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
        type="CosineAnnealingLR",
        T_max=400,
        eta_min=0.00001,
        by_epoch=True,
        begin=30,
        end=400,
)]

# --- 8. 训练、验证、测试配置 ---
train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# --- 9. 运行时默认设置 ---
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 每 5 个 epoch 保存一次 checkpoint。
    # `save_best="auto"` 会根据 `val_evaluator` 的第一个指标（这里是 precision）自动保存最优模型。
    # 我们可以明确指定为 f1-score (如果它在 MultiLabelMetric 中排在前面) 或其他你关心的指标。
    # `rule='greater'` 表示指标值越大越好。
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=10, save_best="auto", rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# 自定义钩子 (用于可视化数据增强效果，可选)
custom_hooks = [
    dict(
        type='VisualizationHook',
        enable=False, # 默认关闭，设为 True 可以在训练时可视化部分样本
        interval=20, # 每 20 个 batch 可视化一次
        show=False,
        draw_gt=True,
        draw_pred=True,
        out_dir='./tmp_out/tongue_shape_visualization', # 可视化结果保存目录
    ),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
# --- 10. 加载预训练权重 ---
# 强烈建议从 ImageNet 预训练的权重开始训练，可以加速收敛并提升性能。
# !!! 请将此路径修改为您下载的 EfficientNetV2-XL 预训练模型路径。
load_from = "/home/an/mmpretrain/works/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth"

# `resume=False` 表示我们是开始一个新任务的训练，而不是从之前的断点恢复。
resume = False
randomness = dict(seed=None, deterministic=False)
