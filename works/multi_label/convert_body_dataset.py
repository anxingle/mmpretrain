#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舌体形态数据集转换脚本
专门用于将 body3_6000_0805_bbox 数据集转换为多标签格式

使用方法：
    python convert_body_dataset.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """执行数据集转换"""
    
    # 设置路径
    script_dir = Path(__file__).parent
    works_dir = script_dir.parent
    datasets_dir = works_dir / "datasets"
    
    # 输入数据集路径
    train_input = datasets_dir / "body3_6000_0805_bbox"
    test_input = datasets_dir / "test_body3_6000_0805_bbox"
    
    # 输出数据集路径
    train_output = datasets_dir / "body3_multilabel_train"
    test_output = datasets_dir / "body3_multilabel_test"
    
    # 检查输入数据集是否存在
    if not train_input.exists():
        print(f"❌ 训练数据集不存在: {train_input}")
        return False
        
    if not test_input.exists():
        print(f"❌ 测试数据集不存在: {test_input}")
        return False
    
    print("🚀 开始转换舌体形态数据集...")
    print(f"📁 训练集: {train_input} → {train_output}")
    print(f"📁 测试集: {test_input} → {test_output}")
    print()
    
    # 转换脚本路径
    convert_script = script_dir / "prepare_multilabel_dataset.py"
    
    # 转换训练集
    print("📊 转换训练集...")
    train_cmd = [
        sys.executable, str(convert_script),
        "--input_dir", str(train_input),
        "--output_dir", str(train_output),
        "--class_names", "teeth", "swift", "crack", "normal",
        "--single_label_ratio", "0.7",
        "--multi_label_ratio", "0.3",
        "--train_ratio", "0.9",
        "--val_ratio", "0.1",
        "--test_ratio", "0.0",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        print("✅ 训练集转换完成")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练集转换失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    # 转换测试集
    print("📊 转换测试集...")
    test_cmd = [
        sys.executable, str(convert_script),
        "--input_dir", str(test_input),
        "--output_dir", str(test_output),
        "--class_names", "teeth", "swift", "crack", "normal",
        "--single_label_ratio", "0.8",
        "--multi_label_ratio", "0.2",
        "--train_ratio", "0.0",
        "--val_ratio", "0.0",
        "--test_ratio", "1.0",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
        print("✅ 测试集转换完成")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试集转换失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    print()
    print("🎉 数据集转换完成！")
    print()
    print("📋 转换结果:")
    print(f"   训练集: {train_output}")
    print(f"   测试集: {test_output}")
    print()
    print("📊 查看统计报告:")
    print(f"   cat {train_output}/statistics_report.json")
    print(f"   cat {test_output}/statistics_report.json")
    print()
    print("🚀 开始训练:")
    print("   cd /home/an/mmpretrain")
    print("   python tools/train.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py")
    
    return True

def check_requirements():
    """检查环境要求"""
    script_dir = Path(__file__).parent
    convert_script = script_dir / "prepare_multilabel_dataset.py"
    
    if not convert_script.exists():
        print(f"❌ 转换脚本不存在: {convert_script}")
        return False
        
    return True

if __name__ == "__main__":
    print("🔧 舌体形态多标签数据集转换工具")
    print("=" * 50)
    
    # 检查环境
    if not check_requirements():
        print("❌ 环境检查失败")
        sys.exit(1)
    
    # 执行转换
    success = main()
    
    if success:
        print("\n✅ 转换成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 转换失败！")
        sys.exit(1)