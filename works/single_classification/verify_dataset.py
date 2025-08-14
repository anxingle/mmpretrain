#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集验证脚本 - 用于检查齿痕舌与正常舌象数据集
"""

import os
import sys
sys.path.insert(0, '/home/an/mmpretrain')

from mmengine.config import Config
from mmpretrain.datasets import build_dataset
import matplotlib.pyplot as plt
import numpy as np

def verify_dataset():
    """验证数据集是否正确加载"""
    
    # 加载配置文件
    cfg = Config.fromfile('teeth_normal_classification.py')
    
    print("=" * 60)
    print("数据集验证")
    print("=" * 60)
    
    # 构建训练数据集
    train_dataset = build_dataset(cfg.train_dataloader.dataset)
    
    print(f"\n✓ 训练数据集路径: {cfg.train_dataloader.dataset.data_root}")
    print(f"✓ 训练集样本数量: {len(train_dataset)}")
    
    # 获取类别信息
    classes = train_dataset.CLASSES if hasattr(train_dataset, 'CLASSES') else None
    if classes:
        print(f"✓ 类别名称: {classes}")
    else:
        # 尝试从文件夹名称获取类别
        data_root = cfg.train_dataloader.dataset.data_root
        if os.path.exists(data_root):
            subdirs = [d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))]
            print(f"✓ 检测到的类别文件夹: {subdirs}")
            
            # 统计每个类别的样本数
            for subdir in subdirs:
                class_dir = os.path.join(data_root, subdir)
                num_samples = len([f for f in os.listdir(class_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  - {subdir}: {num_samples} 张图片")
    
    # 显示几个样本
    print("\n正在可视化前4个训练样本...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx in range(min(4, len(train_dataset))):
        sample = train_dataset[idx]
        
        # 获取图像数据
        if 'img' in sample:
            img = sample['img']
            if isinstance(img, np.ndarray):
                # 如果是numpy数组，检查维度
                if img.ndim == 3 and img.shape[2] == 3:
                    axes[idx].imshow(img)
                elif img.ndim == 3 and img.shape[0] == 3:
                    # CHW格式转换为HWC
                    img = np.transpose(img, (1, 2, 0))
                    axes[idx].imshow(img)
                else:
                    axes[idx].imshow(img, cmap='gray')
            
            # 获取标签
            label = sample.get('gt_label', 'Unknown')
            axes[idx].set_title(f'Sample {idx+1}, Label: {label}')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('works/dataset_samples.png', dpi=100, bbox_inches='tight')
    print(f"✓ 样本可视化已保存到: works/dataset_samples.png")
    
    print("\n" + "=" * 60)
    print("数据集验证完成!")
    print("=" * 60)
    
    # 检查类别平衡
    print("\n类别平衡分析:")
    crack_count = len([f for f in os.listdir(os.path.join(data_root, 'crack')) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
    normal_count = len([f for f in os.listdir(os.path.join(data_root, 'normal')) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    ratio = normal_count / crack_count if crack_count > 0 else 0
    print(f"  齿痕舌(crack): {crack_count} 张")
    print(f"  正常(normal): {normal_count} 张")
    print(f"  类别比例 (normal/crack): {ratio:.2f}")
    print(f"\n建议的类别权重:")
    print(f"  crack/teeth weight: {ratio:.2f}")
    print(f"  normal weight: 1.0")

if __name__ == "__main__":
    verify_dataset()
