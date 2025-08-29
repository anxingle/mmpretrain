# 舌苔颜色分类数据增强策略指南

## 🎯 概述

舌苔颜色分类是一个特殊的医学图像分类任务，需要平衡数据增强的强度与保持舌苔颜色特征的真实性。本文档提供了针对该任务的数据增强策略建议。

## 📊 当前配置分析

### 已使用的方法：
1. ✅ **水平翻转** - 舌体左右对称，安全使用
2. ✅ **颜色抖动** - 轻微调整，保持颜色真实感
3. ✅ **高斯模糊** - 模拟拍摄失焦

### 改进空间：
- 缺乏自动化数据增强
- 空间变换不够丰富
- 缺少批次级增强

## 🚀 推荐的数据增强方法

### 1. 几何变换类

#### 1.1 随机翻转 (RandomFlip)
```python
dict(type='RandomFlip', prob=0.5, direction='horizontal')
```
- **适用性**: ✅ 舌体左右对称，安全使用
- **效果**: 增加数据多样性，提高泛化能力

#### 1.2 随机裁剪 (RandomResizedCrop)
```python
dict(
    type='RandomResizedCrop',
    scale=384,
    crop_ratio_range=(0.85, 1.0),  # 保持85%-100%的内容
    interpolation='lanczos',
    backend='cv2'
)
```
- **适用性**: ⚠️ 需要谨慎使用，保持大部分舌体内容
- **效果**: 提高模型对不同视角的鲁棒性

### 2. 颜色增强类

#### 2.1 颜色抖动 (ColorJitter)
```python
dict(
    type='ColorJitter',
    brightness=0.12,    # 亮度变化
    contrast=0.12,      # 对比度变化
    saturation=0.08,    # 饱和度变化
    hue=0.03,           # 轻微色调变化
    backend='pillow',
)
```
- **适用性**: ✅ 模拟不同光照条件
- **注意事项**: 保持舌苔颜色的相对关系

#### 2.2 对比度均衡化 (Equalize)
```python
dict(type='Equalize', prob=0.15)
```
- **适用性**: ✅ 增强颜色对比度
- **效果**: 突出舌苔颜色差异

#### 2.3 锐化增强 (Sharpness)
```python
dict(type='Sharpness', magnitude=0.4, prob=0.25)
```
- **适用性**: ✅ 增强边缘细节
- **效果**: 提高模型对细节的敏感性

### 3. 自动化数据增强

#### 3.1 RandAugment
```python
dict(
    type='RandAugment',
    policies='timm_increasing',
    num_policies=2,        # 每次随机选择2个策略
    total_level=10,
    magnitude_level=6,     # 中等强度
    magnitude_std=0.5,
    hparams=dict(
        pad_val=[round(x) for x in [123.675, 116.28, 103.53]], 
        interpolation='bicubic'
    )
)
```
- **适用性**: ✅ 自动选择最优增强策略
- **效果**: 提高训练效率和模型性能

#### 3.2 AutoAugment
```python
dict(
    type='AutoAugment',
    policies='imagenet',
    hparams=dict(
        pad_val=[round(x) for x in [123.675, 116.28, 103.53]],
        interpolation='bicubic'
    )
)
```
- **适用性**: ✅ 基于搜索的自动增强
- **效果**: 找到最优的增强策略组合

### 4. 噪声和遮挡类

#### 4.1 高斯模糊 (GaussianBlur)
```python
dict(
    type='GaussianBlur',
    magnitude_range=(0.1, 0.4),
    magnitude_std='inf',
    prob=0.3
)
```
- **适用性**: ✅ 模拟拍摄失焦
- **效果**: 提高模型对模糊图像的鲁棒性

#### 4.2 随机擦除 (RandomErasing)
```python
dict(
    type='RandomErasing',
    erase_prob=0.1,           # 10%概率
    mode='rand',
    min_area_ratio=0.01,      # 最小擦除面积比例
    max_area_ratio=0.1,       # 最大擦除面积比例
    fill_color=[123.675, 116.28, 103.53],  # 使用均值填充
    fill_std=[58.395, 57.12, 57.375]
)
```
- **适用性**: ✅ 模拟遮挡和噪声
- **效果**: 提高模型对部分遮挡的鲁棒性

#### 4.3 Albumentations 噪声增强
```python
dict(
    type='Albu',
    transforms=[
        dict(type='GaussNoise', std_range=(0.02, 0.08), p=0.99),   # 高斯噪声
        # dict(type='ISONoise', color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.0),  # ISO噪声
    ],
    keymap={'img': 'image'}
)
```
- **适用性**: ✅ 模拟相机噪声和传感器噪声
- **效果**: 提高模型对真实相机噪声的鲁棒性
- **安装**: `uv add albumentations`

### 5. 批次级增强

#### 5.1 Mixup
```python
dict(
    type='Mixup',
    alpha=0.2,
    num_classes=3,
    prob=0.3
)
```
- **适用性**: ✅ 在批次内混合不同样本
- **效果**: 提高模型泛化能力

#### 5.2 CutMix
```python
dict(
    type='CutMix',
    alpha=1.0,
    num_classes=3,
    prob=0.3
)
```
- **适用性**: ✅ 在批次内裁剪和混合
- **效果**: 提高模型对局部特征的识别能力

## 🎨 针对舌苔颜色特点的策略

### 1. 颜色保持原则
- **避免过度颜色变换**: 保持舌苔颜色的相对关系
- **适度亮度调整**: 模拟不同光照条件
- **保持对比度**: 确保颜色差异可见

### 2. 形态保持原则
- **避免剧烈形变**: 保持舌体基本形态
- **适度裁剪**: 保持大部分舌体内容
- **边缘保护**: 避免过度模糊边缘

### 3. 医学图像特殊性
- **真实性优先**: 避免产生医学上不合理的图像
- **特征保持**: 保持诊断相关的特征
- **一致性**: 确保增强后的图像仍可用于诊断

## 📈 实验建议

### 1. 渐进式增强
```python
# 第一阶段：基础增强
basic_augmentations = [
    'RandomFlip',
    'ColorJitter',
    'GaussianBlur'
]

# 第二阶段：添加自动化增强
advanced_augmentations = [
    'RandAugment',
    'RandomErasing'
]

# 第三阶段：添加批次级增强
batch_augmentations = [
    'Mixup',
    'CutMix'
]
```

### 2. 消融实验
建议进行以下消融实验：
1. 基础增强 vs 无增强
2. 添加RandAugment的效果
3. 添加批次级增强的效果
4. 不同颜色增强强度的对比

### 3. 性能指标
关注以下指标的变化：
- Top-1 Accuracy
- 各类别的Precision和Recall
- 类别间性能的平衡性
- 模型泛化能力

## 🛠️ 实施建议

### 1. 配置文件选择
- **保守版本**: `coat_color_classification.py` (当前配置)
- **增强版本**: `coat_color_classification_enhanced.py` (推荐配置)

### 2. 训练策略
```bash
# 使用增强配置
python tools/train.py works/coat_cls/coat_color_classification_enhanced.py

# 监控训练过程
./start_monitor_background.sh
```

### 3. 参数调优
根据实验结果调整以下参数：
- `num_policies` in RandAugment
- `magnitude_level` in RandAugment
- `prob` values in various augmentations
- `alpha` values in Mixup/CutMix

## 🎯 预期效果

### 1. 性能提升
- 提高模型泛化能力
- 减少过拟合
- 提高各类别的平衡性

### 2. 鲁棒性增强
- 对光照变化的鲁棒性
- 对拍摄角度的鲁棒性
- 对部分遮挡的鲁棒性

### 3. 实用性提升
- 在实际应用中的表现更稳定
- 对不同设备拍摄的图像适应性更强

## 📝 注意事项

1. **医学图像特殊性**: 确保增强后的图像仍具有医学诊断价值
2. **颜色真实性**: 避免过度改变舌苔颜色特征
3. **训练监控**: 密切关注训练过程中的性能变化
4. **验证测试**: 在独立的测试集上验证增强效果
5. **专家确认**: 必要时请医学专家确认增强策略的合理性
