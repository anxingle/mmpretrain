#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舌体形态多标签数据集准备脚本

功能：
1. 将单标签数据集转换为多标签格式
2. 生成多标签标注文件
3. 数据集统计和验证
4. 支持自定义标签组合规则

使用方法：
python prepare_multilabel_dataset.py --input_dir datasets/body3_6000_0805_bbox --output_dir datasets/body3_multilabel
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple

class MultiLabelDatasetPreparer:
    """多标签数据集准备器"""
    
    def __init__(self, class_names: List[str] = None):
        """
        初始化
        
        Args:
            class_names: 类别名称列表，默认为 ['teeth', 'swift', 'crack', 'normal']
        """
        self.class_names = class_names or ['teeth', 'swift', 'crack', 'normal']
        self.num_classes = len(self.class_names)
        
        # 类别映射：中文名 -> 英文名
        self.class_mapping = {
            '齿痕': 'teeth',
            '点刺': 'swift', 
            '裂纹': 'crack',
            '正常': 'normal',
            'teeth': 'teeth',
            'swift': 'swift',
            'crack': 'crack',
            'normal': 'normal'
        }
        
    def scan_single_label_dataset(self, dataset_dir: str) -> Dict[str, List[str]]:
        """
        扫描单标签数据集结构
        
        Args:
            dataset_dir: 数据集目录路径
            
        Returns:
            字典，键为类别名，值为该类别的图片路径列表
        """
        dataset_path = Path(dataset_dir)
        class_images = defaultdict(list)
        
        print(f"扫描数据集目录: {dataset_dir}")
        
        # 扫描子目录
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                # 映射类别名
                if class_name in self.class_mapping:
                    mapped_name = self.class_mapping[class_name]
                    
                    # 扫描图片文件
                    for img_file in class_dir.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            class_images[mapped_name].append(str(img_file))
                    
                    print(f"  类别 '{class_name}' -> '{mapped_name}': {len(class_images[mapped_name])} 张图片")
                else:
                    print(f"  警告: 未知类别 '{class_name}'，已跳过")
        
        return dict(class_images)
    
    def generate_multilabel_combinations(self, class_images: Dict[str, List[str]], 
                                       combination_rules: Dict = None) -> List[Tuple[str, List[int]]]:
        """
        生成多标签组合
        
        Args:
            class_images: 单标签数据集信息
            combination_rules: 组合规则，定义如何生成多标签样本
            
        Returns:
            列表，每个元素为 (图片路径, 多标签向量)
        """
        if combination_rules is None:
            # 默认组合规则：主要基于单标签，少量多标签组合
            combination_rules = {
                'single_label_ratio': 0.8,  # 80%保持单标签
                'multi_label_ratio': 0.2,   # 20%生成多标签
                'max_labels_per_sample': 3,  # 每个样本最多3个标签
                'normal_exclusive': True,    # 正常类别与其他类别互斥
            }
        
        multilabel_samples = []
        
        # 1. 处理单标签样本
        single_ratio = combination_rules.get('single_label_ratio', 0.8)
        
        for class_name, image_paths in class_images.items():
            class_idx = self.class_names.index(class_name)
            num_single = int(len(image_paths) * single_ratio)
            
            # 选择部分图片作为单标签
            for img_path in image_paths[:num_single]:
                label_vector = [0] * self.num_classes
                label_vector[class_idx] = 1
                multilabel_samples.append((img_path, label_vector))
        
        # 2. 生成多标签样本
        if combination_rules.get('multi_label_ratio', 0) > 0:
            multilabel_samples.extend(
                self._generate_multi_label_samples(class_images, combination_rules)
            )
        
        return multilabel_samples
    
    def _generate_multi_label_samples(self, class_images: Dict[str, List[str]], 
                                    rules: Dict) -> List[Tuple[str, List[int]]]:
        """
        生成多标签样本
        
        Args:
            class_images: 单标签数据集信息
            rules: 组合规则
            
        Returns:
            多标签样本列表
        """
        multi_samples = []
        multi_ratio = rules.get('multi_label_ratio', 0.2)
        max_labels = rules.get('max_labels_per_sample', 3)
        normal_exclusive = rules.get('normal_exclusive', True)
        
        # 计算需要生成的多标签样本数量
        total_images = sum(len(paths) for paths in class_images.values())
        num_multi_samples = int(total_images * multi_ratio)
        
        print(f"生成 {num_multi_samples} 个多标签样本...")
        
        # 定义常见的多标签组合
        common_combinations = [
            ['teeth', 'crack'],      # 齿痕 + 裂纹
            ['teeth', 'swift'],      # 齿痕 + 点刺
            ['swift', 'crack'],      # 点刺 + 裂纹
            ['teeth', 'swift', 'crack'],  # 三种都有
        ]
        
        # 过滤掉不存在的类别组合
        valid_combinations = []
        for combo in common_combinations:
            if all(cls in class_images and len(class_images[cls]) > 0 for cls in combo):
                valid_combinations.append(combo)
        
        if not valid_combinations:
            print("警告: 没有有效的多标签组合")
            return multi_samples
        
        # 生成多标签样本
        for i in range(num_multi_samples):
            # 随机选择一个组合
            combo = np.random.choice(len(valid_combinations))
            selected_classes = valid_combinations[combo]
            
            # 随机选择一个主类别的图片
            primary_class = np.random.choice(selected_classes)
            remaining_images = [img for img in class_images[primary_class] 
                              if not any(img in used for used in [s[0] for s in multi_samples])]
            
            if remaining_images:
                img_path = np.random.choice(remaining_images)
                
                # 生成标签向量
                label_vector = [0] * self.num_classes
                for cls in selected_classes:
                    if cls in self.class_names:
                        cls_idx = self.class_names.index(cls)
                        label_vector[cls_idx] = 1
                
                # 如果包含正常类别且normal_exclusive为True，则移除其他标签
                if normal_exclusive and 'normal' in selected_classes:
                    label_vector = [0] * self.num_classes
                    normal_idx = self.class_names.index('normal')
                    label_vector[normal_idx] = 1
                
                multi_samples.append((img_path, label_vector))
        
        return multi_samples
    
    def create_multilabel_dataset(self, multilabel_samples: List[Tuple[str, List[int]]], 
                                output_dir: str, copy_images: bool = True) -> None:
        """
        创建多标签数据集
        
        Args:
            multilabel_samples: 多标签样本列表
            output_dir: 输出目录
            copy_images: 是否复制图片文件
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建图片目录
        images_dir = output_path / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # 准备标注数据
        annotations = []
        
        print(f"创建多标签数据集到: {output_dir}")
        print(f"总样本数: {len(multilabel_samples)}")
        
        for i, (img_path, label_vector) in enumerate(multilabel_samples):
            img_path = Path(img_path)
            
            # 生成新的文件名
            new_filename = f"{i:06d}_{img_path.stem}{img_path.suffix}"
            new_img_path = images_dir / new_filename
            
            # 复制图片
            if copy_images:
                shutil.copy2(img_path, new_img_path)
            
            # 添加标注
            annotations.append({
                'filename': new_filename,
                'labels': label_vector,
                'label_names': [self.class_names[j] for j, val in enumerate(label_vector) if val == 1],
                'original_path': str(img_path)
            })
        
        # 保存标注文件
        ann_file = output_path / 'annotations.json'
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump({
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'annotations': annotations
            }, f, indent=2, ensure_ascii=False)
        
        print(f"标注文件已保存: {ann_file}")
        
        # 生成统计报告
        self.generate_statistics_report(annotations, output_path / 'statistics.txt')
    
    def generate_statistics_report(self, annotations: List[Dict], report_path: str) -> None:
        """
        生成数据集统计报告
        
        Args:
            annotations: 标注数据
            report_path: 报告文件路径
        """
        # 统计各类别样本数
        class_counts = Counter()
        label_combination_counts = Counter()
        
        for ann in annotations:
            labels = ann['labels']
            label_names = ann['label_names']
            
            # 统计各类别
            for i, val in enumerate(labels):
                if val == 1:
                    class_counts[self.class_names[i]] += 1
            
            # 统计标签组合
            combo_key = '+'.join(sorted(label_names)) if label_names else 'empty'
            label_combination_counts[combo_key] += 1
        
        # 生成报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("舌体形态多标签数据集统计报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总样本数: {len(annotations)}\n")
            f.write(f"类别数: {self.num_classes}\n")
            f.write(f"类别名称: {', '.join(self.class_names)}\n\n")
            
            f.write("各类别样本数统计:\n")
            f.write("-" * 30 + "\n")
            for class_name in self.class_names:
                count = class_counts[class_name]
                percentage = count / len(annotations) * 100
                f.write(f"{class_name:10s}: {count:6d} ({percentage:5.1f}%)\n")
            
            f.write("\n标签组合统计:\n")
            f.write("-" * 30 + "\n")
            for combo, count in label_combination_counts.most_common():
                percentage = count / len(annotations) * 100
                f.write(f"{combo:20s}: {count:6d} ({percentage:5.1f}%)\n")
            
            # 多标签统计
            multi_label_count = sum(1 for ann in annotations if sum(ann['labels']) > 1)
            single_label_count = len(annotations) - multi_label_count
            
            f.write("\n多标签统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"单标签样本: {single_label_count:6d} ({single_label_count/len(annotations)*100:5.1f}%)\n")
            f.write(f"多标签样本: {multi_label_count:6d} ({multi_label_count/len(annotations)*100:5.1f}%)\n")
        
        print(f"统计报告已保存: {report_path}")
    
    def split_dataset(self, annotations_file: str, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            annotations_file: 标注文件路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        np.random.shuffle(annotations)
        
        total = len(annotations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': annotations[:train_end],
            'val': annotations[train_end:val_end],
            'test': annotations[val_end:]
        }
        
        # 保存分割后的标注文件
        base_path = Path(annotations_file).parent
        for split_name, split_annotations in splits.items():
            split_data = {
                'class_names': data['class_names'],
                'num_classes': data['num_classes'],
                'annotations': split_annotations
            }
            
            split_file = base_path / f'{split_name}_annotations.json'
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            
            print(f"{split_name} 集: {len(split_annotations)} 样本 -> {split_file}")

def main():
    parser = argparse.ArgumentParser(description='舌体形态多标签数据集准备工具')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入的单标签数据集目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出的多标签数据集目录')
    parser.add_argument('--class_names', type=str, nargs='+',
                       default=['teeth', 'swift', 'crack', 'normal'],
                       help='类别名称列表')
    parser.add_argument('--single_label_ratio', type=float, default=0.8,
                       help='单标签样本比例')
    parser.add_argument('--multi_label_ratio', type=float, default=0.2,
                       help='多标签样本比例')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--no_copy', action='store_true',
                       help='不复制图片文件，只生成标注')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 创建数据集准备器
    preparer = MultiLabelDatasetPreparer(args.class_names)
    
    # 扫描输入数据集
    class_images = preparer.scan_single_label_dataset(args.input_dir)
    
    if not class_images:
        print("错误: 未找到有效的图片数据")
        return
    
    # 生成多标签组合
    combination_rules = {
        'single_label_ratio': args.single_label_ratio,
        'multi_label_ratio': args.multi_label_ratio,
        'max_labels_per_sample': 3,
        'normal_exclusive': True,
    }
    
    multilabel_samples = preparer.generate_multilabel_combinations(
        class_images, combination_rules
    )
    
    # 创建多标签数据集
    preparer.create_multilabel_dataset(
        multilabel_samples, args.output_dir, copy_images=not args.no_copy
    )
    
    # 分割数据集
    annotations_file = os.path.join(args.output_dir, 'annotations.json')
    preparer.split_dataset(
        annotations_file, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    print("\n多标签数据集准备完成！")
    print(f"输出目录: {args.output_dir}")
    print("\n使用方法:")
    print(f"  训练: python tools/train.py efficientNetV2_xl_body_morphology_multilabel.py")
    print(f"  测试: python tools/test.py efficientNetV2_xl_body_morphology_multilabel.py <checkpoint>")

if __name__ == '__main__':
    main()