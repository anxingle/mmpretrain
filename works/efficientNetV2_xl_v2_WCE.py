_base_ = [
    '../configs/_base_/models/efficientnet_v2/efficientnetv2_s.py',
    # '../configs/_base_/datasets/imagenet_bs32.py',
    '../configs/_base_/schedules/imagenet_bs256.py',
    '../configs/_base_/default_runtime.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNetV2', arch='xl'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4, # 舌色: 绛(779例)/青(178例)/淡红(4205例)/淡白(240例)
        in_channels=1280,
        loss=dict(
            type='CrossEntropyLoss',
            class_weight=[5.4, 23.62, 1.0, 17.52], # 舌色分类的类别权重
            loss_weight=1.0),
        topk=(1,), # 只看第一个的分类是否正确
    ))

# dataset settings
data_preprocessor = dict(
    num_classes=4, # 舌色 白/红/绛/青
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

# 为什么这两行？
# bgr_mean = data_preprocessor['mean'][::-1]
# bgr_std = data_preprocessor['std'][::-1]
# XXX: 1. 舌头图像分类需注意完整性
#      2. 绝对不能有裁切操作；
#      3. 不能改变舌头的颜色
#      4. 不能发生图片的遮挡
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
    #     type='RandomResizedCrop',
    #     scale=192,
    #     backend='pillow',
    #     interpolation='bicubic'), # 这里一定要删掉！舌诊要求完整性
    # XXX: 随机翻转，水平方向
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # XXX: 舌体形态 (舌形 胖/瘦/正常) 可暂时忽略 色彩、亮度、对比度、饱和度 的增强
    # XXX: 慎用 ColorJitter
    dict(
        type='ColorJitter',
        brightness=0.10, # 亮度扰动范围 80%~120% 之间随机调整
        contrast=0.09, # 对比度
        saturation=0.109, # 饱和度
        hue=[0.00001, 0.0099], # 色调偏移范围
        backend='pillow',
    ),
    # XXX: 随机灰度化
    # dict(
    #     type='RandomGrayscale',
    #     prob=0.05,  # 仅5%的概率应用变换
    #     keep_channels=True, # 保持3通道格式
    #     color_format='rgb' # 需要确定图像格式 
    # ),
    # XXX: 随机高斯模糊
    dict(
        type='GaussianBlur',
        magnitude_range=(0.4, 0.7),
        magnitude_std='inf',
        # size_threshold=256,  # 小于此尺寸则减弱模糊
        prob=0.6
    ),
    # XXX: 按照指定边长调整图像尺寸
    dict(
        type='ResizeEdge',
        scale=384,           # 目标边缘长度
        edge='long',         # 指定边缘: 'short'短边/'long'长边
        interpolation='lanczos',  # 插值方法 lanczos精度最高
        backend='cv2'             # 后端实现库
    ),
    dict(
        type='Pad',
        size=(384, 384),
        pad_val=0,
        padding_mode='constant'
    ),
    dict(type='PackInputs'),
]
custom_hooks = [
    # 新增可视化钩子：保存RandAugment增强后的图像
    dict(
        type='VisualizationHook',
        enable=True,
        interval=10,
        show=False,
        draw_gt=True,
        draw_pred=True,
        out_dir='./tmp/512_color_v2_WCE',  # 图像保存目录
        # sample_rate=0.9  # 该参数不再支持
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='long', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="CustomDataset",
        data_root='/home/an/mmpretrain/works/datasets/color6000_0717_cropped_bbox',
        # split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="CustomDataset",
        data_root='/home/an/mmpretrain/works/datasets/test_color6000_0717_cropped_bbox',
        # split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type="CustomDataset",
        data_root='/home/an/mmpretrain/works/datasets/test_color6000_0717_cropped_bbox',
        # split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# learning policy
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[80, 100, 130, 160, 300], gamma=0.1)
param_scheduler = [
    # 预热阶段
    dict(
        type='LinearLR',
        start_factor=0.01,  # 初始学习率为0.1倍
        by_epoch=True,  # 按照epoch进行调度
        begin=0,  # 从第0个epoch开始
        end=30,  # 预热5个epoch
        convert_to_iter_based=True,  # 转换为迭代次数基础的调度
    ),
    # 余弦退火阶段
    dict(
        type="CosineAnnealingLR",
        T_max=400,  # 训练总轮数
        eta_min=0.00001,  # 最小学习率
        by_epoch=True,  # 按照epoch进行调度
        begin=30,  # 从第20个epoch开始
        end=400,  # 训练到第400个epoch结束
)]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=5)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = None
# auto_scale_lr = dict(base_batch_size=256)

# default runtime
# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=10, save_best="auto"),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'
# XXX: 加载预训练模型
# load_from = "/home/an/mmpretrain/works/efficientNetV2_xl_v2_color/epoch_160.pth"
load_from = "/home/an/mmpretrain/works/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth"
# XXX: 这里我们加载预训练权重初始化模型参数，不是恢复上一次的训练进度
# resume = True
resume = False
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
