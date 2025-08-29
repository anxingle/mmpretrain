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
import datetime

def parse_metrics_from_log(log_file):
    """从日志文件中解析最新的评估指标"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 查找最后一次验证的结果
    last_val_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if 'validation' in lines[i].lower() or 'val' in lines[i].lower() or 'accuracy/top1' in lines[i]:
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
    
    # 改进accuracy解析 - 支持多种格式
    for i in range(last_val_idx, min(len(lines), last_val_idx+30)):
        # 尝试多种accuracy格式
        acc_patterns = [
            r'accuracy[/_]top1:\s*([\d.]+)',  # 原格式
            r'accuracy/top1:\s*([\d.]+)',     # mmpretrain常见格式
            r'top1_acc:\s*([\d.]+)',          # 另一种格式
            r'acc.*?(\d+\.?\d*)%?',           # 通用格式
        ]
        
        for pattern in acc_patterns:
            acc_match = re.search(pattern, lines[i], re.IGNORECASE)
            if acc_match:
                acc_value = float(acc_match.group(1))
                # 如果数值大于1，可能是百分比格式，需要除以100
                if acc_value > 1:
                    acc_value = acc_value / 100
                metrics['accuracy'] = acc_value
                break
        if metrics['accuracy'] is not None:
            break
    
    # 改进per-class metrics解析 - 扩大搜索范围
    for i in range(last_val_idx, min(len(lines), last_val_idx+100)):
        line = lines[i]
        
        # 查找classification report或per-class metrics
        if 'precision' in line.lower() and 'recall' in line.lower():
            # 尝试解析类别指标
            class_metrics = parse_class_metrics(lines[i:i+20])
            if class_metrics:
                metrics['per_class'] = class_metrics
                break
        
        # 也尝试查找单独的precision/recall输出
        if any(keyword in line.lower() for keyword in ['precision', 'recall', 'f1-score']):
            class_metrics = parse_class_metrics(lines[i:i+15])
            if class_metrics:
                metrics['per_class'] = class_metrics
                break
    
    return metrics

def parse_class_metrics(lines):
    """解析类别级别的指标 - 支持更多格式"""
    metrics = {}
    
    # 定义类别名称（根据您的任务调整）
    class_names = ['灰色', '白色', '黄色']  # 舌苔颜色分类的类别
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 新增：解析 single-label 格式的数组
        # 格式：single-label/precision_classwise: [50.0, 21.656051635742188, 97.45042419433594]
        precision_match = re.search(r'single-label/precision_classwise:\s*\[([\d.,\s]+)\]', line)
        recall_match = re.search(r'single-label/recall_classwise:\s*\[([\d.,\s]+)\]', line)
        
        if precision_match and recall_match:
            try:
                # 解析数组字符串
                precision_str = precision_match.group(1)
                recall_str = recall_match.group(1)
                
                precision_values = [float(x.strip()) for x in precision_str.split(',')]
                recall_values = [float(x.strip()) for x in recall_str.split(',')]
                
                # 创建类别指标
                for i, (precision, recall) in enumerate(zip(precision_values, recall_values)):
                    if i < len(class_names):
                        class_name = class_names[i]
                    else:
                        class_name = f'Class_{i}'
                    
                    metrics[class_name] = {
                        'precision': precision / 100.0,  # 转换为小数
                        'recall': recall / 100.0
                    }
                break  # 找到后就跳出
            except (ValueError, IndexError) as e:
                print(f"解析数组时出错: {e}")
                continue
            
        # 尝试匹配多种格式的输出
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
        
        # 格式3: sklearn classification report格式
        # class_name    0.xxx    0.xxx    0.xxx    support
        match3 = re.search(r'^(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+', line)
        if match3:
            class_name = match3.group(1)
            if class_name not in ['accuracy', 'macro', 'weighted']:  # 排除汇总行
                metrics[class_name] = {
                    'precision': float(match3.group(2)),
                    'recall': float(match3.group(3))
                }
        
        # 格式4: mmpretrain格式 - class_name/precision: 0.xxx
        precision_match_old = re.search(r'(\w+)/precision:\s*([\d.]+)', line, re.IGNORECASE)
        recall_match_old = re.search(r'(\w+)/recall:\s*([\d.]+)', line, re.IGNORECASE)
        
        if precision_match_old:
            class_name = precision_match_old.group(1)
            if class_name not in metrics:
                metrics[class_name] = {}
            metrics[class_name]['precision'] = float(precision_match_old.group(2))
        
        if recall_match_old:
            class_name = recall_match_old.group(1)
            if class_name not in metrics:
                metrics[class_name] = {}
            metrics[class_name]['recall'] = float(recall_match_old.group(2))
    
    # 过滤掉不完整的指标
    complete_metrics = {}
    for class_name, class_data in metrics.items():
        if 'precision' in class_data and 'recall' in class_data:
            complete_metrics[class_name] = class_data
    
    return complete_metrics if complete_metrics else None

def display_metrics(metrics, task_name, is_first_display=False, last_metrics=None, output_file=None):
    """美化显示评估指标 - 追加模式"""
    output_lines = []
    
    if is_first_display:
        # 只在第一次显示时清屏
        os.system('clear' if os.name != 'nt' else 'cls')
        header = "="*60
        title = f"🔍 训练监控 - {task_name}"
        output_lines.extend([header, title, header])
        print(header)
        print(title)
        print(header)
    else:
        # 非首次显示，添加分隔线
        separator = "\n" + "="*60
        output_lines.append(separator)
        print(separator)
    
    if not metrics:
        if is_first_display:
            waiting_msg = "⏳ 等待第一次验证..."
            output_lines.append(waiting_msg)
            print(waiting_msg)
        return output_lines
    
    # 显示时间戳
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # 检查是否有数据变化
    if last_metrics is not None:
        change_indicator = "🔄 数据更新:" if metrics != last_metrics else "⏸️  数据未变:"
    else:
        change_indicator = "🚀 首次数据:"
    
    timestamp_line = f"\n⏰ [{timestamp}] {change_indicator}"
    output_lines.append(timestamp_line)
    print(timestamp_line)
    
    # 显示基本信息
    if metrics['epoch']:
        epoch_line = f"📊 Epoch: {metrics['epoch']}"
        output_lines.append(epoch_line)
        print(epoch_line)
    
    if metrics['accuracy'] is not None:
        # 如果accuracy有变化，显示变化箭头
        if last_metrics and last_metrics.get('accuracy') is not None:
            acc_change = metrics['accuracy'] - last_metrics['accuracy']
            if acc_change > 0:
                arrow = "📈"
            elif acc_change < 0:
                arrow = "📉"
            else:
                arrow = "➡️"
            acc_line = f"🎯 Top-1 Accuracy: {metrics['accuracy']:.2%} {arrow}"
        else:
            acc_line = f"🎯 Top-1 Accuracy: {metrics['accuracy']:.2%}"
        output_lines.append(acc_line)
        print(acc_line)
    
    # 显示每个类别的指标
    if metrics['per_class']:
        metrics_title = "📈 类别指标:"
        output_lines.append(metrics_title)
        print(metrics_title)
        
        separator = "-"*60
        output_lines.append(separator)
        print(separator)
        
        header = f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}"
        output_lines.append(header)
        print(header)
        
        output_lines.append(separator)
        print(separator)
        
        for class_name, class_metrics in metrics['per_class'].items():
            precision = class_metrics.get('precision', 0)
            recall = class_metrics.get('recall', 0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 检查是否与上次有变化
            change_marker = ""
            if last_metrics and last_metrics.get('per_class') and class_name in last_metrics['per_class']:
                last_class = last_metrics['per_class'][class_name]
                if precision != last_class.get('precision', 0) or recall != last_class.get('recall', 0):
                    change_marker = " ⚡"
            
            class_line = f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}{change_marker}"
            output_lines.append(class_line)
            print(class_line)
        
        # 计算平均值
        if len(metrics['per_class']) > 0:
            avg_precision = sum(m.get('precision', 0) for m in metrics['per_class'].values()) / len(metrics['per_class'])
            avg_recall = sum(m.get('recall', 0) for m in metrics['per_class'].values()) / len(metrics['per_class'])
            avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            output_lines.append(separator)
            print(separator)
            
            avg_line = f"{'Macro Average':<15} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}"
            output_lines.append(avg_line)
            print(avg_line)
    else:
        warning_msg = "⚠️  未找到类别指标 (precision/recall)"
        output_lines.append(warning_msg)
        print(warning_msg)
    
    output_lines.append("-"*60)
    print("-"*60)
    
    # 写入文件
    if output_file:
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
        except Exception as e:
            print(f"⚠️  写入文件失败: {e}")
    
    return output_lines

def monitor_training(log_file, task_name, interval=10, debug=False, clear_mode=False, output_file=None):
    """实时监控训练过程"""
    print(f"开始监控: {log_file}")
    print(f"刷新间隔: {interval}秒")
    
    if debug:
        print("🔧 调试模式已启用")
    
    if clear_mode:
        print("🔄 清屏模式: 每次更新都会清屏")
    else:
        print("📜 追加模式: 历史记录保留在屏幕上")
    
    if output_file:
        print(f"📄 输出文件: {output_file}")
        # 清空文件内容，重新开始
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 训练监控记录 - {task_name}\n")
                f.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"日志文件: {log_file}\n\n")
        except Exception as e:
            print(f"⚠️  初始化文件失败: {e}")
    
    # 记录上一次的指标，用于检测变化
    last_metrics = None
    is_first_display = True
    
    try:
        while True:
            current_metrics = parse_metrics_from_log(log_file)
            
            # 检查是否有新的数据
            if current_metrics != last_metrics and current_metrics is not None:
                # 数据发生了变化，显示新数据
                if clear_mode:
                    # 清屏模式：每次都清屏
                    display_metrics(current_metrics, task_name, True, last_metrics, output_file)
                else:
                    # 追加模式：只在首次清屏
                    display_metrics(current_metrics, task_name, is_first_display, last_metrics, output_file)
                    is_first_display = False
                last_metrics = current_metrics
            elif debug and not current_metrics:
                if is_first_display:
                    print(f"\n🔍 调试信息: 未找到验证指标")
                    print(f"   请检查日志文件: {log_file}")
                    print(f"   确保训练过程中包含 'validation', 'val', 或 'accuracy/top1' 关键词")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        # 写入结束信息
        if output_file:
            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n# 监控结束\n")
                    f.write(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            except Exception as e:
                print(f"⚠️  写入结束信息失败: {e}")
        
        print("\n\n👋 监控已停止")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--task', type=str, default='coat_color',
                       choices=['coat_color'],
                       help='监控的任务类型')
    parser.add_argument('--log', type=str, default=None,
                       help='日志文件路径（默认自动查找）')
    parser.add_argument('--interval', type=int, default=60,
                       help='刷新间隔（秒）')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式，显示更多信息')
    parser.add_argument('--clear', action='store_true',
                       help='清屏模式：每次更新都清屏显示（默认是追加模式）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径（如：recall_acc.md）')
    
    args = parser.parse_args()
    
    # 任务配置
    task_configs = {
        'coat_color': {
            'name': '舌苔颜色分类',
            'default_log': '/data-ssd/logs/coat_color_cls/20250827_120018/20250827_120018.log'
        }
        # 'crack': {
        #     'name': '裂纹分类',
        #     'default_log': '../crack_normal_classification/train.log'
        # },
        # 'swift': {
        #     'name': '点刺分类',
        #     'default_log': '../swift_normal_classification/train.log'
        # }
    }
    
    task_config = task_configs[args.task]
    log_file = args.log or task_config['default_log']
    output_file = args.output
    
    # 检查日志文件
    if not os.path.exists(log_file):
        print(f"⚠️ 日志文件不存在: {log_file}")
        print(f"请先开始训练: bash train_{args.task}_normal.sh")
        sys.exit(1)
    
    # 开始监控
    monitor_training(log_file, task_config['name'], args.interval, args.debug, args.clear, output_file)

if __name__ == "__main__":
    main()
