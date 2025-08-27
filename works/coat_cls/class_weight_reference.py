#!/usr/bin/env python3
"""
舌苔颜色分类 - 类别权重计算参考

数据分布:
- gray: 22 张 (0.40%)
- white: 668 张 (12.07%) 
- yellow: 4844 张 (87.53%)

不平衡比例: 220:1 (极度不平衡)
"""

import torch
import numpy as np

# 类别样本数量
class_counts = [22, 668, 4844]  # [gray, white, yellow]
class_names = ['gray', 'white', 'yellow']

print("舌苔颜色分类 - 推荐的类别权重配置")
print("=" * 60)

# 推荐方案：平方根反比例权重（平衡效果和稳定性）
recommended_weights = [9.16, 1.66, 0.62]  # [gray, white, yellow]

print(f"\n✅ 推荐配置 (平方根反比例权重):")
print(f"class_weight = {recommended_weights}")
print(f"")
print(f"在 MMPretrain 配置文件中使用:")
print(f"```python")
print(f"loss=dict(")
print(f"    type='CrossEntropyLoss',")
print(f"    class_weight={recommended_weights},  # [gray, white, yellow]")
print(f"    loss_weight=1.0")
print(f"),")
print(f"```")

print(f"\n📊 权重解释:")
for i, (name, count, weight) in enumerate(zip(class_names, class_counts, recommended_weights)):
    print(f"  {name:6}: {count:4}张 → 权重 {weight:.2f} (重要性提升 {weight:.1f}倍)")

print(f"\n🔄 其他可选方案:")

# 方案1: 反比例权重（强调少数类）
inverse_weights = [83.85, 2.76, 0.38]
print(f"1. 反比例权重 (更强调少数类): {inverse_weights}")

# 方案2: 温和权重（手动调整）
mild_weights = [5.0, 1.5, 0.8]
print(f"2. 温和权重 (手动调整): {mild_weights}")

# 方案3: 有效样本数方法
ens_weights = [2.89, 0.10, 0.02]
print(f"3. 有效样本数方法: {ens_weights}")

print(f"\n💡 训练建议:")
print(f"1. 从推荐权重开始训练")
print(f"2. 观察验证集上各类别的精度和召回率")
print(f"3. 如果 gray 类召回率太低，可以适当增加其权重 (如 12.0)")
print(f"4. 如果 yellow 类精度下降太多，可以适当增加其权重 (如 0.8)")
print(f"5. 建议配合 Focal Loss 或数据增强技术")

print(f"\n🔧 PyTorch 代码示例:")
print(f"```python")
print(f"import torch")
print(f"import torch.nn as nn")
print(f"")
print(f"# 创建带权重的交叉熵损失")
print(f"class_weights = torch.tensor({recommended_weights})")
print(f"criterion = nn.CrossEntropyLoss(weight=class_weights)")
print(f"```")
