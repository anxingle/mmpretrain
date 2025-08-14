#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单张图片预测脚本 - 用于预测舌象是齿痕舌还是正常
"""

import os
import sys
sys.path.insert(0, '/home/an/mmpretrain')

from mmpretrain import ImageClassificationInferencer
import argparse

def predict_image(image_path, checkpoint_path=None, config_path='teeth_normal_classification.py'):
    """
    对单张图片进行预测
    
    Args:
        image_path: 图片路径
        checkpoint_path: 模型权重路径
        config_path: 配置文件路径
    """
    
    # 如果没有指定checkpoint，使用最新的
    if checkpoint_path is None:
        work_dir = 'works/teeth_normal_classification'
        checkpoints = [f for f in os.listdir(work_dir) if f.endswith('.pth')]
        if checkpoints:
            # 选择最新的checkpoint
            checkpoints.sort()
            checkpoint_path = os.path.join(work_dir, checkpoints[-1])
            print(f"使用checkpoint: {checkpoint_path}")
        else:
            print("错误：未找到训练好的模型权重文件!")
            return
    
    # 创建推理器
    inferencer = ImageClassificationInferencer(
        model=config_path,
        pretrained=checkpoint_path,
        device='cuda:0'  # 如果没有GPU，改为'cpu'
    )
    
    # 进行预测
    result = inferencer(image_path, show=False)
    
    # 解析结果
    pred_class = result['pred_class']
    pred_score = result['pred_score']
    pred_label = result['pred_label']
    
    # 类别映射
    class_names = {
        0: '齿痕舌 (Teeth-marked tongue)',
        1: '正常 (Normal tongue)'
    }
    
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"输入图片: {image_path}")
    print(f"预测类别: {class_names.get(pred_label, 'Unknown')}")
    print(f"置信度: {pred_score:.2%}")
    print("=" * 60)
    
    # 显示所有类别的概率
    print("\n各类别概率:")
    for idx, (cls_name, score) in enumerate(zip(class_names.values(), result['pred_scores'])):
        print(f"  {cls_name}: {score:.2%}")

def main():
    parser = argparse.ArgumentParser(description='舌象分类预测脚本')
    parser.add_argument('image', type=str, help='待预测的图片路径')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='模型权重文件路径（默认使用最新的）')
    parser.add_argument('--config', type=str, 
                       default='teeth_normal_classification.py',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"错误：图片文件不存在: {args.image}")
        sys.exit(1)
    
    predict_image(args.image, args.checkpoint, args.config)

if __name__ == "__main__":
    main()
