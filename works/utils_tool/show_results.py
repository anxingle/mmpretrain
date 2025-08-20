#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强可视化结果展示脚本

该脚本用于展示 visualize_augmentation_rgb.py 生成的结果统计信息
"""

import os
import glob

def show_augmentation_results():
    """展示数据增强可视化结果"""
    print("=" * 60)
    print("数据增强可视化结果统计")
    print("=" * 60)

    # 检查原图目录
    original_dir = "/home/an/mmpretrain/augmentation_samples_rgb/original"
    if os.path.exists(original_dir):
        original_files = glob.glob(os.path.join(original_dir, "*.jpg"))
        print(f"✓ 原图数量: {len(original_files)} 张")
        print(f"  保存路径: {original_dir}")
    else:
        print("✗ 原图目录不存在")

    # 检查增强图目录
    augmented_dir = "/home/an/mmpretrain/augmentation_samples_rgb/augmented"
    if os.path.exists(augmented_dir):
        augmented_files = glob.glob(os.path.join(augmented_dir, "*.jpg"))
        print(f"✓ 增强图数量: {len(augmented_files)} 张")
        print(f"  保存路径: {augmented_dir}")
        if len(original_files) > 0:
            print(f"  平均每张原图生成: {len(augmented_files) // len(original_files)} 个增强版本")
    else:
        print("✗ 增强图目录不存在")

    # 检查对比图目录
    comparison_dir = "/home/an/mmpretrain/comparison_rgb"
    if os.path.exists(comparison_dir):
        comparison_files = glob.glob(os.path.join(comparison_dir, "*.jpg"))
        print(f"✓ 对比图数量: {len(comparison_files)} 张")
        print(f"  保存路径: {comparison_dir}")
    else:
        print("✗ 对比图目录不存在")

    print("\n" + "=" * 60)
    print("数据增强管道包含的变换:")
    print("=" * 60)
    print("1. LoadImageFromFile - 加载图片")
    print("2. ResizeEdge - 边缘缩放")
    print("3. RandomFlip - 随机翻转")
    print("4. ColorJitter - 颜色抖动")
    print("5. GaussianBlur - 高斯模糊")
    print("6. CenterCrop - 中心裁剪")
    print("7. Pad - 填充")

    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("• 原图: 展示经过基础预处理的图片")
    print("• 增强图: 展示应用数据增强后的多个版本")
    print("• 对比图: 并排展示原图和增强图的效果对比")
    print("\n可以使用图片查看器打开这些文件来查看数据增强效果!")
    print("=" * 60)

if __name__ == "__main__":
    show_augmentation_results()