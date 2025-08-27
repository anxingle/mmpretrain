#!/usr/bin/env python3
"""
IPython 中正确使用 MMPretrain 数据集的指南

复制以下代码到您的 IPython 会话中：
"""

# =============================================================================
# 在 IPython 中使用的正确代码
# =============================================================================

# 1. 导入必要的模块
from mmpretrain.registry import DATASETS
from mmpretrain.datasets import CustomDataset

# 2. 定义数据集配置（字典）
dataset_config = dict(
    type='CustomDataset',
    data_root='/data-ssd/coating6000_color/'
)

# 3. 实例化数据集对象（这是关键步骤！）
dataset = DATASETS.build(dataset_config)

# 4. 现在您可以访问数据集的属性了
print(f"数据集类型: {type(dataset)}")
print(f"数据集类别: {dataset.CLASSES}")  
print(f"数据集长度: {len(dataset)}")
print(f"类别数量: {len(dataset.CLASSES)}")  # 注意：没有 num_classes 属性！
print(f"类别映射: {dataset.class_to_idx}")
print(f"数据根目录: {dataset.data_root}")
print(f"元信息: {dataset.metainfo}")

# 5. 获取一个样本
sample = dataset[0]
print(f"样本结构: {list(sample.keys())}")
print(f"图片路径: {sample['img_path']}")
print(f"标签: {sample['gt_label']}")

# 6. 查看数据集的元信息
if hasattr(dataset, 'metainfo'):
    print(f"元信息: {dataset.metainfo}")

# 7. 遍历前几个样本
print("\n前3个样本的信息:")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"样本 {i}: 路径={sample['img_path']}, 标签={sample['gt_label']}")

# =============================================================================
# 如果您想要带数据增强的数据集
# =============================================================================

# 定义数据预处理流水线
from mmpretrain.datasets.transforms import LoadImageFromFile, Resize, CenterCrop, ToTensor, PackInputs

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# 创建带流水线的数据集
dataset_with_pipeline = DATASETS.build(dict(
    type='CustomDataset',
    data_root='/data-ssd/coating6000_color/',
    pipeline=pipeline
))

print(f"\n带流水线的数据集长度: {len(dataset_with_pipeline)}")

# 获取处理后的样本
processed_sample = dataset_with_pipeline[0]
print(f"处理后样本结构: {list(processed_sample.keys())}")
if 'inputs' in processed_sample:
    print(f"图像张量形状: {processed_sample['inputs'].shape}")
