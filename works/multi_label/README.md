# 舌体形态多标签分类解决方案

## 🎯 任务概述

将您的舌体形态单标签数据集转换为多标签格式，并使用 EfficientNetV2-XL 进行多标签分类训练。

**类别**: 齿痕舌(teeth)、点刺舌(swift)、裂纹舌(crack)、正常(normal)

## 🚀 快速开始

### 一键运行（推荐）

```bash
cd /home/an/mmpretrain/works/multi_label
./quick_start.sh
```

这个脚本会自动完成：
1. 数据集格式转换
2. 生成多标签标注文件
3. 启动模型训练

### 分步执行

#### 1. 转换数据集

```bash
cd /home/an/mmpretrain/works/multi_label
python convert_body_dataset.py
```

#### 2. 开始训练

```bash
cd /home/an/mmpretrain
python tools/train.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py
```

#### 3. 测试模型

```bash
python tools/test.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py \
    work_dirs/efficientNetV2_xl_body_morphology_multilabel/latest.pth
```

## 📁 文件结构

```
works/multi_label/
├── README.md                                          # 本文件
├── quick_start.sh                                     # 一键运行脚本
├── convert_body_dataset.py                           # 数据集转换脚本
├── prepare_multilabel_dataset.py                     # 通用多标签转换工具
├── efficientNetV2_xl_body_morphology_multilabel.py   # 训练配置文件
├── dataset_conversion_guide.md                       # 详细转换指南
├── multilabel_config_guide.md                        # 配置文件说明
└── multilabel_example.py                             # 使用示例
```

## 📊 数据集转换

### 输入格式（单标签）
```
body3_6000_0805_bbox/
├── crack/     # 裂纹舌图片
├── normal/    # 正常舌图片
├── swift/     # 点刺舌图片
└── teeth/     # 齿痕舌图片
```

### 输出格式（多标签）
```
body3_multilabel_train/
├── images/                    # 所有图片
├── train_annotations.json     # 训练集标注
├── val_annotations.json       # 验证集标注
└── statistics_report.json    # 统计报告
```

### 多标签示例
- `[1, 0, 0, 0]` - 只有齿痕
- `[0, 1, 1, 0]` - 点刺 + 裂纹
- `[1, 0, 1, 0]` - 齿痕 + 裂纹
- `[0, 0, 0, 1]` - 正常舌体

## 🔧 配置说明

### 模型配置
- **骨干网络**: EfficientNetV2-XL
- **分类头**: MultiLabelLinearClsHead
- **损失函数**: AsymmetricLoss（专为多标签不平衡数据设计）
- **优化器**: AdamW
- **学习率**: 0.0005

### 数据增强
- 随机翻转、旋转
- 颜色抖动（保持纹理清晰）
- 高斯模糊（轻微）
- 高质量缩放和填充

### 评估指标
- MultiLabelMetric（宏平均、微平均）
- AveragePrecision（mAP）

## 📈 训练监控

### TensorBoard
```bash
tensorboard --logdir work_dirs/efficientNetV2_xl_body_morphology_multilabel/
```

### 日志文件
```
work_dirs/efficientNetV2_xl_body_morphology_multilabel/
├── 20241xxx_xxxxxx.log        # 训练日志
├── latest.pth                 # 最新模型
├── best_MultiLabelMetric-mAP_epoch_xx.pth  # 最佳模型
└── vis_data/                  # 可视化结果
```

## 🎛️ 自定义配置

### 调整多标签比例

编辑 `convert_body_dataset.py`：
```python
"--single_label_ratio", "0.7",  # 70% 单标签
"--multi_label_ratio", "0.3",   # 30% 多标签
```

### 调整损失函数

编辑配置文件中的 `loss` 部分：
```python
# AsymmetricLoss（推荐）
loss=dict(
    type='AsymmetricLoss',
    gamma_neg=4,
    gamma_pos=1,
    clip=0.05
)

# 或使用 BCEWithLogitsLoss
loss=dict(
    type='BCEWithLogitsLoss',
    use_sigmoid=True,
    reduction='mean'
)
```

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 降低 `batch_size`
   - 使用更小的模型（如 EfficientNetV2-S）

2. **数据集路径错误**
   - 检查输入目录是否存在
   - 确保目录结构正确

3. **转换失败**
   - 检查磁盘空间
   - 确保有写权限

### 获取帮助

查看详细文档：
- `dataset_conversion_guide.md` - 数据集转换详细说明
- `multilabel_config_guide.md` - 配置文件详细说明
- `multilabel_example.py` - 代码使用示例

## 📝 使用建议

1. **数据备份**: 转换前备份原始数据
2. **渐进训练**: 先用小数据集验证流程
3. **监控指标**: 关注 mAP 和各类别的 F1 分数
4. **模型选择**: 根据显存情况选择合适的模型大小
5. **超参调优**: 根据验证集表现调整学习率和损失函数参数

---

🎉 **现在您可以开始舌体形态多标签分类训练了！**