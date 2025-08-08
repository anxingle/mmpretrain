# 舌体形态多标签数据集转换指南

## 数据集现状

您当前的数据集是**单标签格式**，需要转换为**多标签格式**才能用于 `efficientNetV2_xl_body_morphology_multilabel.py` 训练。

### 当前数据集结构
```
works/datasets/
├── body3_6000_0805_bbox/          # 训练集
│   ├── crack/                      # 裂纹舌图片
│   ├── normal/                     # 正常舌图片
│   ├── swift/                      # 点刺舌图片
│   └── teeth/                      # 齿痕舌图片
└── test_body3_6000_0805_bbox/      # 测试集
    ├── crack/
    ├── normal/
    ├── swift/
    └── teeth/
```

## 转换方法

### 方法一：使用现有转换脚本（推荐）

我们已经为您准备了专门的转换脚本 `prepare_multilabel_dataset.py`。

#### 1. 转换训练集
```bash
cd /home/an/mmpretrain/works/multi_label

python prepare_multilabel_dataset.py \
    --input_dir ../datasets/body3_6000_0805_bbox \
    --output_dir ../datasets/body3_multilabel_train \
    --class_names teeth swift crack normal \
    --single_label_ratio 0.7 \
    --multi_label_ratio 0.3 \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    --test_ratio 0.0 \
    --seed 42
```

#### 2. 转换测试集
```bash
python prepare_multilabel_dataset.py \
    --input_dir ../datasets/test_body3_6000_0805_bbox \
    --output_dir ../datasets/body3_multilabel_test \
    --class_names teeth swift crack normal \
    --single_label_ratio 0.8 \
    --multi_label_ratio 0.2 \
    --train_ratio 0.0 \
    --val_ratio 0.0 \
    --test_ratio 1.0 \
    --seed 42
```

### 参数说明

- `--input_dir`: 输入的单标签数据集目录
- `--output_dir`: 输出的多标签数据集目录
- `--class_names`: 类别名称列表（teeth, swift, crack, normal）
- `--single_label_ratio`: 保持单标签的样本比例（0.7 = 70%）
- `--multi_label_ratio`: 生成多标签的样本比例（0.3 = 30%）
- `--train_ratio`: 训练集比例
- `--val_ratio`: 验证集比例
- `--test_ratio`: 测试集比例
- `--seed`: 随机种子，确保结果可复现

### 转换后的数据集结构

```
works/datasets/
├── body3_multilabel_train/         # 多标签训练集
│   ├── images/                     # 所有图片文件
│   ├── annotations.json            # 完整标注文件
│   ├── train_annotations.json      # 训练集标注
│   ├── val_annotations.json        # 验证集标注
│   └── statistics_report.json     # 数据集统计报告
└── body3_multilabel_test/          # 多标签测试集
    ├── images/
    ├── annotations.json
    ├── test_annotations.json
    └── statistics_report.json
```

### 多标签标注格式

转换后的标注文件采用 MMPretrain 标准格式：

```json
{
    "metainfo": {
        "classes": ["teeth", "swift", "crack", "normal"]
    },
    "data_list": [
        {
            "img_path": "images/1.png",
            "gt_label": [1, 0, 1, 0]  // 同时有齿痕和裂纹
        },
        {
            "img_path": "images/2.png", 
            "gt_label": [0, 1, 0, 0]  // 只有点刺
        },
        {
            "img_path": "images/3.png",
            "gt_label": [0, 0, 0, 1]  // 正常舌体
        }
    ]
}
```

## 更新配置文件

转换完成后，需要更新 `efficientNetV2_xl_body_morphology_multilabel.py` 中的数据集路径：

```python
# 训练数据加载器
train_dataloader = dict(
    batch_size=6,
    num_workers=8,
    dataset=dict(
        type="MultiLabelDataset",  # 改为多标签数据集
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_train',
        ann_file='train_annotations.json',  # 指定标注文件
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# 验证数据加载器
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type="MultiLabelDataset",
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_train',
        ann_file='val_annotations.json',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# 测试数据加载器
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type="MultiLabelDataset",
        data_root='/home/an/mmpretrain/works/datasets/body3_multilabel_test',
        ann_file='test_annotations.json',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
```

## 多标签生成策略

脚本会根据以下策略生成多标签样本：

### 1. 单标签保持（70%）
- 直接将原始单标签转换为多标签格式
- `crack/1.png` → `[0, 0, 1, 0]`（只有裂纹）
- `normal/2.png` → `[0, 0, 0, 1]`（正常）

### 2. 多标签组合（30%）
- **常见组合**：
  - 齿痕 + 裂纹：`[1, 0, 1, 0]`
  - 点刺 + 裂纹：`[0, 1, 1, 0]`
  - 齿痕 + 点刺：`[1, 1, 0, 0]`
- **排除规则**：
  - 正常舌体不与其他异常组合
  - 最多同时3种异常特征

## 验证转换结果

转换完成后，检查生成的统计报告：

```bash
# 查看训练集统计
cat ../datasets/body3_multilabel_train/statistics_report.json

# 查看测试集统计
cat ../datasets/body3_multilabel_test/statistics_report.json
```

统计报告包含：
- 总样本数
- 各类别分布
- 多标签组合统计
- 标签共现矩阵

## 开始训练

数据集转换完成后，即可开始训练：

```bash
cd /home/an/mmpretrain

# 训练模型
python tools/train.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py

# 测试模型
python tools/test.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py \
    work_dirs/efficientNetV2_xl_body_morphology_multilabel/latest.pth
```

## 注意事项

1. **数据备份**：转换前请备份原始数据集
2. **存储空间**：转换过程会复制图片，确保有足够存储空间
3. **标注质量**：多标签标注比单标签更复杂，建议人工抽查验证
4. **类别平衡**：注意观察转换后的类别分布，必要时调整组合比例
5. **随机种子**：使用相同的随机种子确保结果可复现

## 故障排除

### 常见问题

1. **内存不足**：使用 `--no_copy` 参数只生成标注文件，不复制图片
2. **路径错误**：确保输入路径存在且包含正确的子目录结构
3. **权限问题**：确保对输出目录有写权限

### 自定义转换

如需自定义转换规则，可以修改 `prepare_multilabel_dataset.py` 中的 `combination_rules` 参数：

```python
combination_rules = {
    'single_label_ratio': 0.6,      # 调整单标签比例
    'multi_label_ratio': 0.4,       # 调整多标签比例
    'max_labels_per_sample': 2,     # 限制每个样本最多标签数
    'normal_exclusive': True,       # 正常舌体是否排他
    'preferred_combinations': [     # 优先生成的组合
        ['teeth', 'crack'],         # 齿痕+裂纹
        ['swift', 'crack']          # 点刺+裂纹
    ]
}
```

这样就完成了从单标签到多标签数据集的转换，可以开始进行舌体形态多标签分类训练了！