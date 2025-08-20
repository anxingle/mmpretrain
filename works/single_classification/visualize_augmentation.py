#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强效果可视化脚本
用于查看 train_pipeline 中各种数据增强操作的效果
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def get_demo_images():
    """获取demo目录中的示例图片"""
    demo_dir = Path('/home/an/mmpretrain/demo')
    image_files = []
    
    # 查找demo目录中的图片文件
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG']:
        image_files.extend(demo_dir.glob(ext))
    
    return [str(img) for img in image_files]

def load_image(image_path):
    """加载图片"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def random_flip_horizontal(img, prob=0.5):
    """水平翻转"""
    if random.random() < prob:
        return cv2.flip(img, 1)
    return img

def color_jitter(img, brightness=0.15, contrast=0.15, saturation=0.159, hue=0.0099):
    """颜色抖动"""
    # 转换为PIL图像
    pil_img = Image.fromarray(img)
    
    # 亮度调整
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_factor)
    
    # 对比度调整
    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_factor)
    
    # 饱和度调整
    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        pil_img = ImageEnhance.Color(pil_img).enhance(saturation_factor)
    
    # 转换回numpy数组
    return np.array(pil_img)

def gaussian_blur(img, magnitude_range=(0.2, 0.45), prob=0.45):
    """高斯模糊"""
    if random.random() < prob:
        sigma = random.uniform(*magnitude_range)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return img

def resize_edge(img, scale=384, edge='long'):
    """调整图像尺寸（保持长宽比）"""
    h, w = img.shape[:2]
    
    if edge == 'long':
        # 长边缩放到指定尺寸
        if h > w:
            new_h = scale
            new_w = int(w * scale / h)
        else:
            new_w = scale
            new_h = int(h * scale / w)
    else:
        # 短边缩放到指定尺寸
        if h < w:
            new_h = scale
            new_w = int(w * scale / h)
        else:
            new_w = scale
            new_h = int(h * scale / w)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def pad_to_square(img, size=(384, 384), pad_val=0):
    """填充到正方形"""
    h, w = img.shape[:2]
    target_h, target_w = size
    
    # 计算填充量
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    
    # 计算上下左右的填充量
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 填充
    if len(img.shape) == 3:
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                   cv2.BORDER_CONSTANT, value=[pad_val, pad_val, pad_val])
    else:
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                   cv2.BORDER_CONSTANT, value=pad_val)
    
    return padded

def apply_single_augmentation(image_path):
    """应用一次完整的数据增强"""
    # 1. 加载图片
    img = load_image(image_path)
    
    # 2. 水平翻转
    img = random_flip_horizontal(img, prob=0.5)
    
    # 3. 颜色抖动
    img = color_jitter(img, brightness=0.15, contrast=0.15, saturation=0.159, hue=0.0099)
    
    # 4. 高斯模糊
    img = gaussian_blur(img, magnitude_range=(0.2, 0.45), prob=0.45)
    
    # 5. 尺寸调整
    img = resize_edge(img, scale=384, edge='long')
    
    # 6. 填充到正方形
    img = pad_to_square(img, size=(384, 384), pad_val=0)
    
    return img

def visualize_multiple_augmentations(image_path, num_augmentations=8):
    """可视化同一张图片的多次增强效果"""
    # 加载原始图片
    original_img = load_image(image_path)
    
    # 生成多个增强版本
    augmented_images = []
    for i in range(num_augmentations):
        aug_img = apply_single_augmentation(image_path)
        augmented_images.append(aug_img)
    
    # 创建可视化图表
    cols = 3
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    # 显示原图
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('原始图片', fontsize=12, fontweight='bold', color='red')
    axes[0, 0].axis('off')
    
    # 显示增强后的图片
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    for i, (aug_img, pos) in enumerate(zip(augmented_images, positions)):
        row, col = pos
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(f'增强版本 {i+1}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.suptitle(f'数据增强效果展示 - {Path(image_path).name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def analyze_augmentation_components(image_path):
    """分析各个增强组件的单独效果"""
    # 加载原始图片
    original_img = load_image(image_path)
    
    # 应用各个组件
    results = []
    
    # 1. 原图
    results.append(('原图', original_img))
    
    # 2. 水平翻转
    flipped_img = random_flip_horizontal(original_img.copy(), prob=1.0)
    results.append(('水平翻转', flipped_img))
    
    # 3. 颜色抖动
    color_jittered_img = color_jitter(original_img.copy(), brightness=0.15, contrast=0.15, 
                                     saturation=0.159, hue=0.0099)
    results.append(('颜色抖动', color_jittered_img))
    
    # 4. 高斯模糊
    blurred_img = gaussian_blur(original_img.copy(), magnitude_range=(0.2, 0.45), prob=1.0)
    results.append(('高斯模糊', blurred_img))
    
    # 5. 尺寸调整
    resized_img = resize_edge(original_img.copy(), scale=384, edge='long')
    results.append(('尺寸调整', resized_img))
    
    # 6. 填充
    resized_img_for_pad = resize_edge(original_img.copy(), scale=384, edge='long')
    padded_img = pad_to_square(resized_img_for_pad, size=(384, 384), pad_val=0)
    results.append(('填充到正方形', padded_img))
    
    # 创建可视化
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    
    for i, (name, img) in enumerate(results):
        row = i // cols
        col = i % cols
        
        if row < rows and col < cols:
            axes[row, col].imshow(img)
            axes[row, col].set_title(name, fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
    
    plt.suptitle(f'各增强组件效果分析 - {Path(image_path).name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    """主函数"""
    print("=== 数据增强效果可视化工具 ===")
    
    # 获取demo图片
    print("1. 查找demo图片...")
    demo_images = get_demo_images()
    
    if not demo_images:
        print("错误：未找到demo图片")
        return
    
    print(f"找到 {len(demo_images)} 张demo图片: {[Path(img).name for img in demo_images]}")
    
    # 输出目录
    output_dir = '/home/an/mmpretrain/works/single_classification/augmentation_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每张demo图片生成可视化
    for i, image_path in enumerate(demo_images[:2]):  # 只处理前2张图片
        print(f"\n2. 处理图片 {i+1}: {Path(image_path).name}")
        
        try:
            # 生成多次增强效果对比
            print("   - 生成多次增强效果对比...")
            fig1 = visualize_multiple_augmentations(image_path)
            output_path1 = os.path.join(output_dir, f'multiple_augmentations_{Path(image_path).stem}.png')
            fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            print(f"   - 保存到: {output_path1}")
            
            # 生成各组件效果分析
            print("   - 生成各组件效果分析...")
            fig2 = analyze_augmentation_components(image_path)
            output_path2 = os.path.join(output_dir, f'component_analysis_{Path(image_path).stem}.png')
            fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"   - 保存到: {output_path2}")
            
        except Exception as e:
            print(f"   - 处理图片时出错: {str(e)}")
            continue
    
    print(f"\n=== 完成！结果保存在: {output_dir} ===")
    print("\n可视化文件说明:")
    print("- multiple_augmentations_*.png : 同一张图片的多次增强效果对比")
    print("- component_analysis_*.png : 各个增强组件的单独效果分析")
    
    print("\n数据增强管道包含以下步骤:")
    print("1. 水平翻转 (50%概率)")
    print("2. 颜色抖动 (亮度±15%, 对比度±15%, 饱和度±15.9%, 色调微调)")
    print("3. 高斯模糊 (45%概率, 轻度模糊保护细节)")
    print("4. 尺寸调整 (长边缩放到384像素)")
    print("5. 填充到正方形 (384x384)")
    
    print("\n注意事项:")
    print("- 这些增强操作专门为舌体点刺检测任务设计")
    print("- 颜色抖动参数较小，避免影响舌体颜色特征")
    print("- 高斯模糊程度很轻，保护点刺细节特征")
    print("- 使用填充而非裁切，避免丢失重要信息")

if __name__ == '__main__':
    main()