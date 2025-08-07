# 舌体形态多标签分类配置指南

## 配置文件说明

### `efficientNetV2_xl_body_morphology_multilabel.py`

这是一个专门为舌体形态多标签分类任务设计的配置文件，基于您的需求进行了优化。

#### 任务特点
- **多标签分类**: 一个舌体可能同时具有多种形态特征
- **四个类别**: 齿痕舌(teeth)、点刺舌(swift)、裂纹舌(crack)、正常(normal)
- **标签格式**: 每个样本用4位二进制向量表示，如 [1,0,1,0] 表示同时有齿痕和裂纹

#### 模型配置
- **模型**: EfficientNetV2-XL
- **输入尺寸**: 384x384
- **分类头**: MultiLabelLinearClsHead
- **损失函数**: AsymmetricLoss（专为多标签不平衡数据设计）

#### 数据要求

##### 标注格式
每个图像需要多标签标注，格式为 `[teeth, swift, crack, normal]`：
- `[1, 0, 1, 0]` - 同时有齿痕和裂纹
- `[0, 1, 0, 0]` - 只有点刺
- `[0, 0, 0, 1]` - 正常舌体
- `[1, 1, 1, 0]` - 同时有齿痕、点刺、裂纹

##### 数据集结构
```
datasets/
├── body3_6000_0805_bbox/          # 训练集
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── test_body3_6000_0805_bbox/     # 测试集
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

#### 关键特性

##### 1. 损失函数优化
- **AsymmetricLoss**: 专为多标签不平衡数据设计
- **参数调优**: gamma_neg=4, gamma_pos=1, clip=0.05
- **备选方案**: BCEWithLogitsLoss（可根据需要切换）

##### 2. 数据增强策略
- **保持完整性**: 禁用RandomResizedCrop
- **边缘保护**: 适度几何变换，保持形态特征
- **纹理保持**: 轻微模糊，保持纹理清晰度
- **对比度增强**: 突出纹理细节

##### 3. 评估指标
- **mAP**: mean Average Precision（主要指标）
- **Macro F1**: 宏平均F1分数
- **Micro F1**: 微平均F1分数
- **每类别指标**: 精确率、召回率、F1分数

##### 4. 训练优化
- **学习率**: 0.0005（较小，多标签任务更敏感）
- **优化器**: AdamW with weight_decay=0.05
- **调度器**: 线性预热 + 余弦退火
- **正则化**: 增加drop_path_rate=0.2

## 使用方法

### 1. 准备数据
确保数据集按照上述结构组织，并准备多标签标注文件。

### 2. 训练模型
```bash
python tools/train.py efficientNetV2_xl_body_morphology_multilabel.py
```

### 3. 测试模型
```bash
python tools/test.py efficientNetV2_xl_body_morphology_multilabel.py work_dirs/efficientNetV2_xl_body_morphology_multilabel/latest.pth
```

## 与单标签分类的区别

| 特性 | 单标签分类 | 多标签分类 |
|------|------------|------------|
| 输出 | 单个类别 | 多个类别概率 |
| 损失函数 | CrossEntropyLoss | AsymmetricLoss/BCEWithLogitsLoss |
| 评估指标 | Accuracy | mAP, F1-score |
| 标注格式 | 单个标签 | 多标签向量 |
| 应用场景 | 互斥分类 | 可重叠分类 |

## 注意事项

1. **数据标注质量**: 多标签标注比单标签更复杂，需要仔细检查
2. **类别平衡**: 注意各类别的分布，必要时调整损失函数权重
3. **阈值选择**: 推理时需要选择合适的阈值来确定正负样本
4. **计算资源**: EfficientNetV2-XL模型较大，需要足够的GPU内存
5. **训练时间**: 多标签任务通常需要更长的训练时间

## 故障排除

### 常见问题
1. **内存不足**: 降低batch_size或使用梯度累积
2. **收敛慢**: 调整学习率或使用不同的损失函数
3. **类别不平衡**: 调整损失函数的权重参数
4. **过拟合**: 增加正则化或数据增强

### 性能优化建议
1. **数据增强**: 根据实际数据特点调整增强策略
2. **损失函数**: 尝试不同的多标签损失函数
3. **模型大小**: 可以尝试更小的模型（如B0）进行快速实验
4. **学习率调度**: 根据训练曲线调整学习率策略