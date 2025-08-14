# 齿痕舌与正常舌象二分类训练指南

## 项目说明
本项目用于训练一个舌象二分类模型，能够区分齿痕舌(teeth-marked tongue)和正常舌象(normal tongue)。

## 文件说明

### 1. 配置文件
- `teeth_normal_classification.py`: 主要训练配置文件
  - 使用EfficientNetV2-XL作为backbone
  - 配置了适合舌象分类的数据增强策略
  - 设置了类别权重以处理数据不平衡

### 2. 训练脚本
- `train_teeth_normal.sh`: 训练启动脚本
  ```bash
  # 单GPU训练
  bash train_teeth_normal.sh
  ```

### 3. 辅助脚本
- `verify_dataset.py`: 数据集验证脚本
  ```bash
  cd /home/an/mmpretrain/works
  python verify_dataset.py
  ```

- `predict_single_image.py`: 单张图片预测脚本
  ```bash
  # 预测单张图片
  python predict_single_image.py path/to/image.png
  
  # 指定特定的模型权重
  python predict_single_image.py path/to/image.png --checkpoint path/to/checkpoint.pth
  ```

## 数据集信息

### 数据集路径
`/home/an/mmpretrain/works/datasets/crack_VS_normal_0812/`

### 数据分布
- crack (齿痕舌): 2037张图片
- normal (正常): 3467张图片
- 类别比例: 1:1.7

### 类别权重设置
为了处理类别不平衡，配置文件中设置了：
- crack/teeth权重: 1.7
- normal权重: 1.0

## 训练配置要点

### 数据增强策略
根据舌诊图像特点，配置了以下增强：

1. **允许的增强**:
   - ✅ 水平翻转 (prob=0.5)
   - ✅ 高斯模糊 (轻度，prob=0.3)
   - ✅ 尺寸调整保持长宽比 (ResizeEdge)
   - ✅ 填充到正方形 (Pad)

2. **禁用的增强**:
   - ❌ 颜色抖动 (ColorJitter)
   - ❌ 亮度/对比度/饱和度调整
   - ❌ 随机裁切 (RandomResizedCrop)
   - ❌ 图像遮挡 (RandomErasing)

### 模型架构
- Backbone: EfficientNetV2-XL
- 预训练权重: ImageNet预训练
- 输入尺寸: 384×384
- 类别数: 2

### 训练参数
- 优化器: AdamW
- 初始学习率: 0.001
- 训练轮数: 200 epochs
- Batch size: 16
- 学习率策略: 10轮预热 + 余弦退火

## 快速开始

### 1. 验证数据集
```bash
cd /home/an/mmpretrain/works
python verify_dataset.py
```

### 2. 开始训练
```bash
# 方式1: 使用shell脚本
bash train_teeth_normal.sh

# 方式2: 直接使用python
cd /home/an/mmpretrain
python tools/train.py works/teeth_normal_classification.py
```

### 3. 监控训练
训练日志和模型权重将保存在:
```
works/teeth_normal_classification/
├── *.log                 # 训练日志
├── epoch_*.pth           # 模型检查点
├── best_*.pth            # 最佳模型
└── vis_teeth_normal/     # 可视化结果
```

### 4. 模型预测
训练完成后，使用以下命令进行预测：
```bash
cd /home/an/mmpretrain/works
python predict_single_image.py /path/to/test/image.png
```

## 注意事项

1. **显存要求**: 建议使用至少8GB显存的GPU
2. **数据格式**: 确保图片为PNG或JPG格式
3. **预训练权重**: 配置文件已指定预训练权重路径，请确保文件存在
4. **类别文件夹**: 数据集应按照crack/normal文件夹组织

## 可能的优化方向

1. **调整类别权重**: 如果发现模型偏向某一类，可调整class_weight
2. **修改batch_size**: 根据GPU显存调整
3. **增加数据增强**: 如需要可适当增加其他增强策略
4. **早停策略**: 可添加EarlyStopping避免过拟合

## 问题排查

如遇到问题，请检查：
1. 数据集路径是否正确
2. 预训练权重文件是否存在
3. CUDA和PyTorch版本兼容性
4. 数据集图片格式是否正确
