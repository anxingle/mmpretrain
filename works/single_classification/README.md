# 舌象二分类任务训练指南

## 项目概述
本项目包含三个舌象形态特征的二分类任务：
1. **齿痕舌分类**: 齿痕舌(teeth) vs 正常(normal)
2. **裂纹分类**: 裂纹(crack) vs 正常(normal)  
3. **点刺分类**: 点刺(swift) vs 正常(normal)

## 目录结构
```
single_classification/
├── 配置文件/
│   ├── teeth_normal_classification.py  # 齿痕舌分类配置
│   ├── crack_normal_classification.py  # 裂纹分类配置
│   └── swift_normal_classification.py  # 点刺分类配置
├── 训练脚本/
│   ├── train_teeth_normal.sh          # 训练齿痕舌分类
│   ├── train_crack_normal.sh          # 训练裂纹分类
│   ├── train_swift_normal.sh          # 训练点刺分类
│   └── train_all.sh                   # 批量训练所有模型
├── 工具脚本/
│   ├── verify_all_datasets.py         # 数据集验证
│   └── predict_image.py               # 图片预测
└── README.md                           # 本文档
```

## 数据集信息

### 数据集路径
- 齿痕舌: `/home/an/mmpretrain/works/datasets/teeth_VS_normal_0812`
- 裂纹: `/home/an/mmpretrain/works/datasets/crack_VS_normal_0812`
- 点刺: `/home/an/mmpretrain/works/datasets/swift_VS_normal_0812`

### 数据分布
| 分类任务 | 特征类别 | 正常类别 | 总计 | 类别比例 | 建议权重 |
|---------|---------|---------|------|---------|----------|
| 齿痕舌 | 2101 | 3403 | 5504 | 1:1.62 | [1.62, 1.0] |
| 裂纹 | 2037 | 3467 | 5504 | 1:1.70 | [1.70, 1.0] |
| 点刺 | 2159 | 3345 | 5504 | 1:1.55 | [1.55, 1.0] |

## 快速开始

### 1. 验证数据集
在训练前验证所有数据集是否正确：
```bash
cd /home/an/mmpretrain/works/single_classification
python verify_all_datasets.py
```

### 2. 训练模型

#### 训练单个模型
```bash
# 训练齿痕舌分类
bash train_teeth_normal.sh

# 训练裂纹分类
bash train_crack_normal.sh

# 训练点刺分类
bash train_swift_normal.sh
```

#### 批量训练所有模型
```bash
# 依次训练所有三个模型
bash train_all.sh
```

### 3. 模型预测

#### 单张图片预测
```bash
# 使用齿痕舌模型预测（默认）
python predict_image.py path/to/image.png

# 指定使用裂纹模型
python predict_image.py path/to/image.png --task crack

# 指定使用点刺模型
python predict_image.py path/to/image.png --task swift

# 使用所有模型进行预测
python predict_image.py path/to/image.png --task all
```

#### 指定模型权重
```bash
python predict_image.py path/to/image.png \
    --task crack \
    --checkpoint ../crack_normal_classification/best_accuracy_top1_epoch_50.pth
```

## 模型配置详情

### 共同配置
- **Backbone**: EfficientNetV2-XL
- **预训练权重**: ImageNet预训练
- **输入尺寸**: 384×384
- **优化器**: AdamW (lr=0.001)
- **训练轮数**: 200 epochs
- **Batch size**: 16
- **学习率策略**: 10轮预热 + 余弦退火

### 数据增强策略
根据舌诊图像特点，采用了保守的增强策略：

#### ✅ 启用的增强
1. **水平翻转**: 50%概率
2. **高斯模糊**: 轻度模糊，保护纹理细节
   - 齿痕舌: 30%概率，强度0.3-0.5
   - 裂纹: 20%概率，强度0.2-0.4
   - 点刺: 15%概率，强度0.2-0.35
3. **尺寸调整**: ResizeEdge保持长宽比
4. **填充**: Pad到384×384

#### ❌ 禁用的增强
- 颜色抖动 (ColorJitter)
- 亮度/对比度/饱和度调整
- 随机裁切 (RandomResizedCrop)
- 图像遮挡 (RandomErasing)

## 训练输出

训练过程中会生成以下文件：
```
works/
├── teeth_normal_classification/    # 齿痕舌模型
├── crack_normal_classification/    # 裂纹模型
├── swift_normal_classification/    # 点刺模型
│   ├── *.log                      # 训练日志
│   ├── epoch_*.pth                # 各轮次检查点
│   ├── best_*.pth                 # 最佳模型
│   └── vis_*/                     # 可视化结果
```

## 性能监控

### ✅ 是的！会显示每个类别的Precision/Recall

配置文件已更新为使用`SingleLabelMetric`评估器，训练过程中会显示：
- **每个类别**的Precision、Recall、F1-Score
- **整体**的Accuracy
- **宏平均**和**加权平均**指标

### 实时监控训练
```bash
# 使用监控脚本查看实时指标（推荐）
python monitor_training.py --task teeth  # 监控齿痕舌分类
python monitor_training.py --task crack  # 监控裂纹分类
python monitor_training.py --task swift  # 监控点刺分类

# 或者查看原始日志
tail -f ../teeth_normal_classification/train.log

# 查看损失曲线
grep "loss:" ../teeth_normal_classification/train.log | tail -20
```

### 评估指标示例输出
```
验证结果 - Epoch [50/200]:
----------------------------------------
Top-1 Accuracy: 92.35%

类别指标:
├─ teeth类:  Precision=0.916, Recall=0.897, F1=0.906
├─ normal类: Precision=0.932, Recall=0.949, F1=0.940
└─ 宏平均:   Precision=0.924, Recall=0.923, F1=0.923
```

### TensorBoard可视化
如果安装了TensorBoard，可以可视化训练过程：
```bash
tensorboard --logdir ../teeth_normal_classification
```

## 常见问题

### 1. 显存不足
- 减小batch_size（在配置文件中修改）
- 使用更小的模型（如EfficientNetV2-S或M）

### 2. 训练速度慢
- 启用cudnn_benchmark（在env_cfg中设置为True）
- 使用多GPU训练（修改训练脚本）

### 3. 过拟合
- 增加数据增强
- 减小模型复杂度
- 添加早停机制

### 4. 类别不平衡严重
- 调整class_weight参数
- 使用focal loss替代cross entropy loss
- 采用过采样或欠采样策略

## 进阶优化

### 1. 超参数调优
- 学习率: 尝试1e-4到1e-2之间
- Weight decay: 尝试0.001到0.1之间
- Batch size: 根据显存调整

### 2. 模型架构
- 尝试其他backbone: ResNet, ConvNeXt, Swin Transformer
- 调整模型大小: S, M, L, XL

### 3. 数据增强
- MixUp/CutMix: 提升泛化能力
- AutoAugment: 自动搜索最佳增强策略
- Test Time Augmentation: 提升预测准确率

### 4. 损失函数
- Focal Loss: 处理类别不平衡
- Label Smoothing: 防止过拟合
- ArcFace/CosFace: 增强特征判别性

## 联系与支持

如有问题，请查看：
1. MMPretrain官方文档: https://mmpretrain.readthedocs.io/
2. 项目日志文件: works/*/train.log
3. 数据集验证结果: python verify_all_datasets.py

## 更新日志

- 2024.08.12: 初始版本，支持三种舌象形态分类
  - 齿痕舌 vs 正常
  - 裂纹 vs 正常
  - 点刺 vs 正常
