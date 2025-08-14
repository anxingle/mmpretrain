#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试配置文件是否可以正确加载
"""

import os
import sys
sys.path.insert(0, '/home/an/mmpretrain')

from mmengine.config import Config

def test_config(config_file):
    """测试单个配置文件"""
    print(f"\n测试配置文件: {config_file}")
    print("-" * 40)
    
    try:
        cfg = Config.fromfile(config_file)
        
        # 检查关键配置
        print(f"✓ 模型类型: {cfg.model.type}")
        print(f"✓ Backbone: {cfg.model.backbone.type} - {cfg.model.backbone.arch}")
        print(f"✓ 类别数: {cfg.model.head.num_classes}")
        print(f"✓ 类别权重: {cfg.model.head.loss.class_weight}")
        print(f"✓ 数据集路径: {cfg.train_dataloader.dataset.data_root}")
        print(f"✓ Batch size: {cfg.train_dataloader.batch_size}")
        print(f"✓ 训练轮数: {cfg.train_cfg.max_epochs}")
        print(f"✓ 优化器: {cfg.optim_wrapper.optimizer.type}")
        print(f"✓ 学习率: {cfg.optim_wrapper.optimizer.lr}")
        
        print("\n配置文件测试通过！✅")
        return True
        
    except Exception as e:
        print(f"\n❌ 配置文件加载失败: {e}")
        return False

def main():
    """测试所有配置文件"""
    print("=" * 60)
    print("配置文件测试")
    print("=" * 60)
    
    configs = [
        'teeth_normal_classification.py',
        'crack_normal_classification.py',
        'swift_normal_classification.py'
    ]
    
    results = []
    for config in configs:
        if os.path.exists(config):
            success = test_config(config)
            results.append((config, success))
        else:
            print(f"\n❌ 配置文件不存在: {config}")
            results.append((config, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for config, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{config:<40} {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有配置文件测试通过！可以开始训练。")
        print("\n下一步：")
        print("  1. 验证数据集: python verify_all_datasets.py")
        print("  2. 开始训练: bash train_all.sh")
    else:
        print("\n⚠️ 部分配置文件有问题，请检查后再训练。")

if __name__ == "__main__":
    main()

