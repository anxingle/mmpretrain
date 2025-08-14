#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练监控脚本 - 实时显示各类别的precision/recall
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path

def parse_metrics_from_log(log_file):
    """从日志文件中解析最新的评估指标"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 查找最后一次验证的结果
    last_val_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if 'validation' in lines[i].lower() or 'val' in lines[i].lower():
            last_val_idx = i
            break
    
    if last_val_idx == -1:
        return None
    
    # 提取该次验证的指标
    metrics = {
        'epoch': None,
        'accuracy': None,
        'per_class': {},
        'loss': None
    }
    
    # 查找epoch信息
    for i in range(last_val_idx, max(0, last_val_idx-20), -1):
        epoch_match = re.search(r'Epoch\s*\[(\d+)/(\d+)\]', lines[i])
        if epoch_match:
            metrics['epoch'] = f"{epoch_match.group(1)}/{epoch_match.group(2)}"
            break
    
    # 查找accuracy
    for i in range(last_val_idx, min(len(lines), last_val_idx+30)):
        acc_match = re.search(r'accuracy[/_]top1:\s*([\d.]+)', lines[i], re.IGNORECASE)
        if acc_match:
            metrics['accuracy'] = float(acc_match.group(1))
            break
    
    # 查找per-class metrics
    for i in range(last_val_idx, min(len(lines), last_val_idx+50)):
        # 匹配precision/recall/f1-score
        if 'precision' in lines[i].lower():
            # 尝试解析类别指标
            class_metrics = parse_class_metrics(lines[i:i+10])
            if class_metrics:
                metrics['per_class'] = class_metrics
                break
    
    return metrics

def parse_class_metrics(lines):
    """解析类别级别的指标"""
    metrics = {}
    
    for line in lines:
        # 尝试匹配不同格式的输出
        # 格式1: class_name precision: 0.xxx recall: 0.xxx
        match1 = re.search(r'(\w+)\s+precision:\s*([\d.]+)\s+recall:\s*([\d.]+)', line, re.IGNORECASE)
        if match1:
            class_name = match1.group(1)
            metrics[class_name] = {
                'precision': float(match1.group(2)),
                'recall': float(match1.group(3))
            }
        
        # 格式2: class_name: precision=0.xxx recall=0.xxx
        match2 = re.search(r'(\w+):\s*precision=([\d.]+)\s*recall=([\d.]+)', line, re.IGNORECASE)
        if match2:
            class_name = match2.group(1)
            metrics[class_name] = {
                'precision': float(match2.group(2)),
                'recall': float(match2.group(3))
            }
    
    return metrics if metrics else None

def display_metrics(metrics, task_name):
    """美化显示评估指标"""
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("="*60)
    print(f"🔍 训练监控 - {task_name}")
    print("="*60)
    
    if not metrics:
        print("⏳ 等待第一次验证...")
        return
    
    # 显示基本信息
    if metrics['epoch']:
        print(f"📊 Epoch: {metrics['epoch']}")
    
    if metrics['accuracy'] is not None:
        print(f"🎯 Top-1 Accuracy: {metrics['accuracy']:.2%}")
    
    # 显示每个类别的指标
    if metrics['per_class']:
        print("\n📈 类别指标:")
        print("-"*60)
        print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*60)
        
        for class_name, class_metrics in metrics['per_class'].items():
            precision = class_metrics.get('precision', 0)
            recall = class_metrics.get('recall', 0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        # 计算平均值
        if len(metrics['per_class']) > 0:
            avg_precision = sum(m.get('precision', 0) for m in metrics['per_class'].values()) / len(metrics['per_class'])
            avg_recall = sum(m.get('recall', 0) for m in metrics['per_class'].values()) / len(metrics['per_class'])
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            print("-"*60)
            print(f"{'Macro Average':<15} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    
    print("\n" + "="*60)
    print("💡 提示: Ctrl+C 退出监控")

def monitor_training(log_file, task_name, interval=10):
    """实时监控训练过程"""
    print(f"开始监控: {log_file}")
    print(f"刷新间隔: {interval}秒")
    
    try:
        while True:
            metrics = parse_metrics_from_log(log_file)
            display_metrics(metrics, task_name)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--task', type=str, default='teeth',
                       choices=['teeth', 'crack', 'swift'],
                       help='监控的任务类型')
    parser.add_argument('--log', type=str, default=None,
                       help='日志文件路径（默认自动查找）')
    parser.add_argument('--interval', type=int, default=10,
                       help='刷新间隔（秒）')
    
    args = parser.parse_args()
    
    # 任务配置
    task_configs = {
        'teeth': {
            'name': '齿痕舌分类',
            'default_log': '../teeth_normal_classification/train.log'
        },
        'crack': {
            'name': '裂纹分类',
            'default_log': '../crack_normal_classification/train.log'
        },
        'swift': {
            'name': '点刺分类',
            'default_log': '../swift_normal_classification/train.log'
        }
    }
    
    task_config = task_configs[args.task]
    log_file = args.log or task_config['default_log']
    
    # 检查日志文件
    if not os.path.exists(log_file):
        print(f"⚠️ 日志文件不存在: {log_file}")
        print(f"请先开始训练: bash train_{args.task}_normal.sh")
        sys.exit(1)
    
    # 开始监控
    monitor_training(log_file, task_config['name'], args.interval)

if __name__ == "__main__":
    main()
