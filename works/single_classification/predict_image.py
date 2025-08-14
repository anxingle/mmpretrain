#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用舌象分类预测脚本 - 支持多种分类任务
"""

import os
import sys
sys.path.insert(0, '/home/an/mmpretrain')

from mmpretrain import ImageClassificationInferencer
import argparse

# 定义各分类任务的配置
TASK_CONFIGS = {
    'teeth': {
        'config': 'teeth_normal_classification.py',
        'work_dir': '../teeth_normal_classification',
        'classes': {0: '齿痕舌 (Teeth-marked)', 1: '正常 (Normal)'},
        'name': '齿痕舌分类'
    },
    'crack': {
        'config': 'crack_normal_classification.py',
        'work_dir': '../crack_normal_classification',
        'classes': {0: '裂纹 (Cracked)', 1: '正常 (Normal)'},
        'name': '裂纹分类'
    },
    'swift': {
        'config': 'swift_normal_classification.py',
        'work_dir': '../swift_normal_classification',
        'classes': {0: '点刺 (Swift)', 1: '正常 (Normal)'},
        'name': '点刺分类'
    }
}

def predict_image(image_path, task='teeth', checkpoint_path=None):
    """
    对单张图片进行预测
    
    Args:
        image_path: 图片路径
        task: 分类任务类型 ('teeth', 'crack', 'swift')
        checkpoint_path: 模型权重路径
    """
    
    if task not in TASK_CONFIGS:
        print(f"错误：不支持的任务类型 '{task}'")
        print(f"支持的任务: {list(TASK_CONFIGS.keys())}")
        return
    
    task_config = TASK_CONFIGS[task]
    config_path = task_config['config']
    
    # 如果没有指定checkpoint，使用最新的
    if checkpoint_path is None:
        work_dir = task_config['work_dir']
        if os.path.exists(work_dir):
            checkpoints = [f for f in os.listdir(work_dir) if f.endswith('.pth')]
            if checkpoints:
                # 选择最新的checkpoint
                checkpoints.sort()
                checkpoint_path = os.path.join(work_dir, checkpoints[-1])
                print(f"使用checkpoint: {checkpoint_path}")
            else:
                print(f"错误：未找到 {task_config['name']} 的训练模型！")
                print(f"请先运行: bash train_{task}_normal.sh")
                return
        else:
            print(f"错误：工作目录不存在: {work_dir}")
            print(f"请先训练 {task_config['name']} 模型")
            return
    
    # 创建推理器
    try:
        inferencer = ImageClassificationInferencer(
            model=config_path,
            pretrained=checkpoint_path,
            device='cuda:0'  # 如果没有GPU，会自动降级到CPU
        )
    except Exception as e:
        print(f"创建推理器失败: {e}")
        print("尝试使用CPU...")
        inferencer = ImageClassificationInferencer(
            model=config_path,
            pretrained=checkpoint_path,
            device='cpu'
        )
    
    # 进行预测
    result = inferencer(image_path, show=False)
    
    # 解析结果
    pred_label = result['pred_label']
    pred_score = result['pred_score']
    
    # 获取类别名称
    class_names = task_config['classes']
    
    print("\n" + "=" * 60)
    print(f"{task_config['name']} - 预测结果")
    print("=" * 60)
    print(f"输入图片: {image_path}")
    print(f"预测类别: {class_names.get(pred_label, 'Unknown')}")
    print(f"置信度: {pred_score:.2%}")
    print("=" * 60)
    
    # 显示所有类别的概率
    if 'pred_scores' in result and len(result['pred_scores']) > 0:
        print("\n各类别概率:")
        for idx, score in enumerate(result['pred_scores']):
            cls_name = class_names.get(idx, f'Class {idx}')
            print(f"  {cls_name}: {score:.2%}")

def batch_predict(image_path):
    """使用所有可用模型进行批量预测"""
    print("\n" + "=" * 60)
    print("批量预测模式 - 使用所有可用模型")
    print("=" * 60)
    
    for task in TASK_CONFIGS.keys():
        try:
            predict_image(image_path, task)
        except Exception as e:
            print(f"\n{TASK_CONFIGS[task]['name']} 预测失败: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='舌象分类预测脚本')
    parser.add_argument('image', type=str, help='待预测的图片路径')
    parser.add_argument('--task', type=str, default='teeth',
                       choices=['teeth', 'crack', 'swift', 'all'],
                       help='分类任务类型 (默认: teeth)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型权重文件路径（默认使用最新的）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"错误：图片文件不存在: {args.image}")
        sys.exit(1)
    
    if args.task == 'all':
        batch_predict(args.image)
    else:
        predict_image(args.image, args.task, args.checkpoint)

if __name__ == "__main__":
    main()
