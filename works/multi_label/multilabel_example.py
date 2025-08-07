#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舌体形态多标签分类使用示例

这个脚本展示了如何使用新创建的多标签分类配置进行训练和推理。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加mmpretrain路径
sys.path.insert(0, '/home/an/mmpretrain')

from mmpretrain import get_model
from mmpretrain.apis import inference_model, init_model
from mmengine import Config

class TongueMultiLabelClassifier:
    """舌体形态多标签分类器"""
    
    def __init__(self, config_file: str, checkpoint_file: str = None):
        """
        初始化分类器
        
        Args:
            config_file: 配置文件路径
            checkpoint_file: 模型权重文件路径
        """
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = None
        self.class_names = ['齿痕舌', '点刺舌', '裂纹舌', '正常']
        self.threshold = 0.5  # 多标签分类阈值
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            self.load_model()
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.checkpoint_file}")
        self.model = init_model(self.config_file, self.checkpoint_file, device='cuda')
        print("模型加载完成")
    
    def predict_single_image(self, image_path: str, threshold: float = None) -> dict:
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            threshold: 分类阈值
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model() 或提供 checkpoint_file")
        
        if threshold is None:
            threshold = self.threshold
        
        # 推理
        result = inference_model(self.model, image_path)
        
        # 解析结果
        if hasattr(result, 'pred_scores'):
            scores = result.pred_scores.cpu().numpy()
        else:
            scores = result['pred_scores']
        
        # 应用阈值
        predictions = (scores > threshold).astype(int)
        
        # 获取预测的类别
        predicted_classes = [self.class_names[i] for i, pred in enumerate(predictions) if pred == 1]
        
        return {
            'image_path': image_path,
            'scores': scores.tolist(),
            'predictions': predictions.tolist(),
            'predicted_classes': predicted_classes,
            'threshold': threshold
        }
    
    def predict_batch(self, image_paths: list, threshold: float = None) -> list:
        """
        批量预测
        
        Args:
            image_paths: 图片路径列表
            threshold: 分类阈值
            
        Returns:
            预测结果列表
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict_single_image(img_path, threshold)
                results.append(result)
            except Exception as e:
                print(f"预测失败 {img_path}: {e}")
                results.append(None)
        
        return results
    
    def evaluate_predictions(self, predictions: list, ground_truth: list) -> dict:
        """
        评估预测结果
        
        Args:
            predictions: 预测结果列表
            ground_truth: 真实标签列表
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import classification_report, multilabel_confusion_matrix
        from sklearn.metrics import average_precision_score, f1_score
        
        # 转换为numpy数组
        y_true = np.array(ground_truth)
        y_pred = np.array([pred['predictions'] for pred in predictions if pred is not None])
        y_scores = np.array([pred['scores'] for pred in predictions if pred is not None])
        
        # 计算各种指标
        metrics = {}
        
        # F1分数
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # 平均精度
        metrics['map'] = average_precision_score(y_true, y_scores, average='macro')
        
        # 每个类别的指标
        class_report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        metrics['per_class'] = class_report
        
        # 混淆矩阵
        cm = multilabel_confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def print_prediction_summary(self, predictions: list):
        """
        打印预测结果摘要
        
        Args:
            predictions: 预测结果列表
        """
        print("\n=== 预测结果摘要 ===")
        
        # 统计各类别预测数量
        class_counts = {name: 0 for name in self.class_names}
        total_predictions = len([p for p in predictions if p is not None])
        
        for pred in predictions:
            if pred is not None:
                for class_name in pred['predicted_classes']:
                    class_counts[class_name] += 1
        
        print(f"总预测样本数: {total_predictions}")
        print("\n各类别预测统计:")
        for class_name, count in class_counts.items():
            percentage = count / total_predictions * 100 if total_predictions > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # 多标签统计
        multi_label_count = sum(1 for pred in predictions 
                               if pred is not None and len(pred['predicted_classes']) > 1)
        single_label_count = sum(1 for pred in predictions 
                                if pred is not None and len(pred['predicted_classes']) == 1)
        no_label_count = sum(1 for pred in predictions 
                            if pred is not None and len(pred['predicted_classes']) == 0)
        
        print("\n标签数量统计:")
        print(f"  单标签: {single_label_count} ({single_label_count/total_predictions*100:.1f}%)")
        print(f"  多标签: {multi_label_count} ({multi_label_count/total_predictions*100:.1f}%)")
        print(f"  无标签: {no_label_count} ({no_label_count/total_predictions*100:.1f}%)")

def demo_training():
    """演示训练过程"""
    print("=== 舌体形态多标签分类训练演示 ===")
    
    config_file = '/home/an/mmpretrain/works/efficientNetV2_xl_body_morphology_multilabel.py'
    
    print(f"配置文件: {config_file}")
    print("\n训练命令:")
    print(f"python tools/train.py {config_file}")
    
    print("\n训练参数说明:")
    print("- 模型: EfficientNetV2-XL")
    print("- 输入尺寸: 384x384")
    print("- 批次大小: 6 (可根据GPU内存调整)")
    print("- 学习率: 0.0005")
    print("- 训练轮数: 200")
    print("- 损失函数: AsymmetricLoss")
    
    print("\n注意事项:")
    print("1. 确保数据集已按多标签格式准备")
    print("2. 根据GPU内存调整batch_size")
    print("3. 监控训练过程中的mAP指标")
    print("4. 可以尝试不同的阈值来优化性能")

def demo_inference():
    """演示推理过程"""
    print("\n=== 舌体形态多标签分类推理演示 ===")
    
    config_file = '/home/an/mmpretrain/works/efficientNetV2_xl_body_morphology_multilabel.py'
    
    # 注意：这里需要实际的模型权重文件
    checkpoint_file = 'work_dirs/efficientNetV2_xl_body_morphology_multilabel/latest.pth'
    
    print(f"配置文件: {config_file}")
    print(f"模型权重: {checkpoint_file}")
    
    if not os.path.exists(checkpoint_file):
        print("\n注意: 模型权重文件不存在，请先训练模型")
        print("这里仅演示推理代码结构")
        
        # 创建分类器实例（不加载模型）
        classifier = TongueMultiLabelClassifier(config_file)
        
        print("\n推理代码示例:")
        print("""
# 创建分类器
classifier = TongueMultiLabelClassifier(config_file, checkpoint_file)

# 单张图片预测
result = classifier.predict_single_image('path/to/image.jpg')
print(f"预测类别: {result['predicted_classes']}")
print(f"置信度: {result['scores']}")

# 批量预测
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = classifier.predict_batch(image_paths)

# 打印预测摘要
classifier.print_prediction_summary(results)
        """)
        
        return
    
    # 如果模型存在，进行实际推理演示
    try:
        classifier = TongueMultiLabelClassifier(config_file, checkpoint_file)
        
        # 查找测试图片
        test_dir = '/home/an/mmpretrain/works/datasets/test_body3_6000_0805_bbox'
        if os.path.exists(test_dir):
            # 获取一些测试图片
            test_images = []
            for root, dirs, files in os.walk(test_dir):
                for file in files[:5]:  # 只取前5张
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append(os.path.join(root, file))
            
            if test_images:
                print(f"\n找到 {len(test_images)} 张测试图片")
                
                # 批量预测
                results = classifier.predict_batch(test_images)
                
                # 显示结果
                for i, result in enumerate(results):
                    if result:
                        print(f"\n图片 {i+1}: {os.path.basename(result['image_path'])}")
                        print(f"  预测类别: {result['predicted_classes']}")
                        print(f"  置信度: {[f'{s:.3f}' for s in result['scores']]}")
                
                # 打印摘要
                classifier.print_prediction_summary(results)
            else:
                print("未找到测试图片")
        else:
            print(f"测试目录不存在: {test_dir}")
            
    except Exception as e:
        print(f"推理演示失败: {e}")

def demo_data_preparation():
    """演示数据准备过程"""
    print("\n=== 数据准备演示 ===")
    
    input_dir = '/home/an/mmpretrain/works/datasets/body3_6000_0805_bbox'
    output_dir = '/home/an/mmpretrain/works/datasets/body3_multilabel'
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    print("\n数据准备命令:")
    print(f"python prepare_multilabel_dataset.py \\")
    print(f"    --input_dir {input_dir} \\")
    print(f"    --output_dir {output_dir} \\")
    print(f"    --single_label_ratio 0.8 \\")
    print(f"    --multi_label_ratio 0.2 \\")
    print(f"    --train_ratio 0.8 \\")
    print(f"    --val_ratio 0.1 \\")
    print(f"    --test_ratio 0.1")
    
    print("\n参数说明:")
    print("- single_label_ratio: 单标签样本比例 (0.8 = 80%)")
    print("- multi_label_ratio: 多标签样本比例 (0.2 = 20%)")
    print("- train/val/test_ratio: 训练/验证/测试集比例")
    
    print("\n输出文件:")
    print("- images/: 图片文件")
    print("- annotations.json: 完整标注文件")
    print("- train_annotations.json: 训练集标注")
    print("- val_annotations.json: 验证集标注")
    print("- test_annotations.json: 测试集标注")
    print("- statistics.txt: 数据集统计报告")

def main():
    """主函数"""
    print("舌体形态多标签分类完整使用示例")
    print("=" * 50)
    
    # 演示数据准备
    demo_data_preparation()
    
    # 演示训练
    demo_training()
    
    # 演示推理
    demo_inference()
    
    print("\n=== 完整工作流程 ===")
    print("1. 准备多标签数据集")
    print("   python prepare_multilabel_dataset.py --input_dir ... --output_dir ...")
    print("\n2. 训练多标签模型")
    print("   python tools/train.py efficientNetV2_xl_body_morphology_multilabel.py")
    print("\n3. 测试模型性能")
    print("   python tools/test.py efficientNetV2_xl_body_morphology_multilabel.py <checkpoint>")
    print("\n4. 使用模型进行推理")
    print("   python multilabel_example.py")
    
    print("\n详细文档请参考: multilabel_config_guide.md")

if __name__ == '__main__':
    main()