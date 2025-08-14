#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集验证脚本 - 验证所有舌象分类数据集
"""

import os
import sys
sys.path.insert(0, '/home/an/mmpretrain')

def verify_dataset(dataset_name, data_root, class_names):
    """验证单个数据集"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"路径: {data_root}")
    print(f"{'='*60}")
    
    if not os.path.exists(data_root):
        print(f"❌ 错误：数据集路径不存在！")
        return
    
    # 统计各类别样本数
    total_samples = 0
    class_counts = {}
    
    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
            count = len(files)
            class_counts[class_name] = count
            total_samples += count
            print(f"  ✓ {class_name}: {count} 张图片")
        else:
            print(f"  ❌ {class_name}: 文件夹不存在")
            class_counts[class_name] = 0
    
    # 计算类别平衡
    if len(class_counts) == 2 and all(v > 0 for v in class_counts.values()):
        counts = list(class_counts.values())
        ratio = max(counts) / min(counts)
        print(f"\n类别平衡分析:")
        print(f"  总样本数: {total_samples}")
        print(f"  类别比例: 1:{ratio:.2f}")
        
        # 建议的类别权重
        if counts[0] < counts[1]:  # 第一类样本少
            weight1 = counts[1] / counts[0]
            weight2 = 1.0
        else:  # 第二类样本少
            weight1 = 1.0
            weight2 = counts[0] / counts[1]
        
        print(f"\n建议的类别权重:")
        print(f"  {class_names[0]}: {weight1:.2f}")
        print(f"  {class_names[1]}: {weight2:.2f}")
    
    return class_counts

def main():
    """验证所有数据集"""
    print("="*60)
    print("舌象分类数据集验证工具")
    print("="*60)
    
    # 定义所有数据集
    datasets = [
        {
            'name': '齿痕舌 vs 正常',
            'root': '/home/an/mmpretrain/works/datasets/teeth_VS_normal_0812',
            'classes': ['teeth', 'normal']
        },
        {
            'name': '裂纹 vs 正常',
            'root': '/home/an/mmpretrain/works/datasets/crack_VS_normal_0812',
            'classes': ['crack', 'normal']
        },
        {
            'name': '点刺 vs 正常',
            'root': '/home/an/mmpretrain/works/datasets/swift_VS_normal_0812',
            'classes': ['swift', 'normal']
        }
    ]
    
    # 验证每个数据集
    all_stats = {}
    for dataset in datasets:
        stats = verify_dataset(
            dataset['name'],
            dataset['root'],
            dataset['classes']
        )
        all_stats[dataset['name']] = stats
    
    # 汇总报告
    print(f"\n{'='*60}")
    print("数据集汇总")
    print(f"{'='*60}")
    print(f"{'数据集':<20} {'类别1':<15} {'类别2':<15} {'总计':<10}")
    print("-"*60)
    
    for dataset in datasets:
        name = dataset['name']
        if name in all_stats:
            stats = all_stats[name]
            if stats:
                class1 = f"{dataset['classes'][0]}: {stats.get(dataset['classes'][0], 0)}"
                class2 = f"{dataset['classes'][1]}: {stats.get(dataset['classes'][1], 0)}"
                total = sum(stats.values())
                print(f"{name:<20} {class1:<15} {class2:<15} {total:<10}")
    
    print(f"\n{'='*60}")
    print("验证完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
