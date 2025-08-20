# MMPretrain 数据增强可视化工具

## 概述

本工具包含了一套完整的数据增强可视化脚本，用于帮助理解和调试 MMPretrain 中的数据增强管道。

## 文件说明

### 主要脚本

1. **`visualize_augmentation_rgb.py`** - 主要的数据增强可视化脚本
   - 生成原图和增强图的对比样本
   - 创建并排对比图
   - 支持完整的 MMPretrain 数据增强管道

2. **`show_results.py`** - 结果统计展示脚本
   - 统计生成的图片数量
   - 展示数据增强管道信息
   - 提供使用说明

## 支持的数据增强变换

本工具支持以下 MMPretrain 数据增强变换：

- **LoadImageFromFile** - 图片加载
- **ResizeEdge** - 边缘缩放（MMPretrain 特有）
- **RandomFlip** - 随机翻转
- **ColorJitter** - 颜色抖动（亮度、对比度、饱和度、色调）
- **GaussianBlur** - 高斯模糊
- **CenterCrop** - 中心裁剪
- **Pad** - 图片填充

## 使用方法

### 1. 运行数据增强可视化

```bash
cd /home/an/mmpretrain
python works/utils_tool/visualize_augmentation_rgb.py
```

### 2. 查看结果统计

```bash
python works/utils_tool/show_results.py
```

## 输出结果

脚本运行后会生成以下目录和文件：

### 目录结构
```
/home/an/mmpretrain/
├── augmentation_samples_rgb/
│   ├── original/          # 原图（经过基础预处理）
│   │   ├── xxx_original.jpg
│   │   └── ...
│   └── augmented/         # 增强图（应用数据增强后）
│       ├── xxx_aug_1.jpg
│       ├── xxx_aug_2.jpg
│       └── ...
└── comparison_rgb/        # 对比图（原图和增强图并排显示）
    ├── xxx_comparison_rgb.jpg
    └── ...
```

### 生成的图片类型

1. **原图** (`*_original.jpg`)
   - 经过基础预处理的图片（ResizeEdge + Pad）
   - 用于对比数据增强效果

2. **增强图** (`*_aug_*.jpg`)
   - 应用完整数据增强管道的图片
   - 每张原图生成 5 个不同的增强版本
   - 展示随机性数据增强的多样性

3. **对比图** (`*_comparison_rgb.jpg`)
   - 原图和增强图的并排对比
   - 便于直观比较增强效果

## 配置说明

脚本会自动读取以下配置文件：
```python
config_file = '/home/an/mmpretrain/works/single_classification/swift_normal_classification.py'
```

如需使用其他配置文件，请修改脚本中的 `config_file` 变量。

## 技术细节

### 变换注册机制

本工具解决了 MMPretrain 中变换注册的问题：

1. **导入策略**：
   - 从 `mmcv.transforms` 导入通用变换
   - 从 `mmpretrain.datasets.transforms` 导入特定变换
   - 确保所有变换正确注册到注册表

2. **管道构建**：
   - 直接实例化变换对象而非使用配置字典
   - 避免注册表查找问题
   - 支持动态参数传递

### 图片处理

- **格式转换**：自动处理 BGR 到 RGB 的转换
- **尺寸标准化**：统一输出 384x384 像素
- **质量保证**：使用高质量 JPEG 保存（quality=95）

## 故障排除

### 常见问题

1. **变换未注册错误**
   ```
   KeyError: 'ResizeEdge is not in the mmengine::transform registry'
   ```
   **解决方案**：确保正确导入了 mmpretrain 模块和相关变换

2. **路径不存在错误**
   ```
   FileNotFoundError: 数据集路径不存在
   ```
   **解决方案**：检查配置文件中的数据集路径是否正确

3. **环境问题**
   ```
   ModuleNotFoundError: No module named 'mmpretrain'
   ```
   **解决方案**：确保在正确的 conda 环境中运行（openmmlab）

### 检查清单

- [ ] 确认在 `openmmlab` 环境中运行
- [ ] 确认 mmpretrain 正确安装
- [ ] 确认配置文件路径正确
- [ ] 确认数据集路径存在
- [ ] 确认有足够的磁盘空间保存结果

## 扩展功能

### 添加新的数据增强变换

要支持新的变换，需要在两个函数中添加相应的处理逻辑：

```python
# 在 save_augmentation_samples_rgb 和 create_side_by_side_comparison_rgb 中添加
elif transform_cfg['type'] == 'NewTransform':
    augmented_transforms.append(NewTransform(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
```

### 自定义输出参数

可以修改脚本中的以下参数：
- `num_samples`：生成的样本数量
- `num_augmented_versions`：每个样本的增强版本数
- `save_dir`：输出目录名称

## 总结

本工具提供了一个完整的解决方案来可视化 MMPretrain 的数据增强效果，帮助用户：

- 理解数据增强管道的工作原理
- 调试数据增强参数
- 评估增强效果的合理性
- 优化训练数据的质量

通过直观的图片对比，用户可以更好地理解和优化自己的数据增强策略。