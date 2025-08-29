#!/bin/bash

# 数据增强可视化脚本
# 自动运行所有可视化配置并生成对比图

echo "🎨 开始数据增强可视化..."

# 创建输出目录
mkdir -p visualization_results

# 1. 基础配置可视化
echo "📊 1. 生成基础配置可视化..."
python tools/visualization/browse_dataset.py works/coat_cls/coat_color_classification.py \
    --phase train \
    --mode pipeline \
    --show-number 5 \
    --output-dir visualization_results/baseline \
    --channel-order BGR \
    --not-show

# 2. 增强配置可视化
echo "📊 2. 生成增强配置可视化..."
python tools/visualization/browse_dataset.py works/coat_cls/coat_color_classification_enhanced.py \
    --phase train \
    --mode pipeline \
    --show-number 5 \
    --output-dir visualization_results/enhanced \
    --channel-order BGR \
    --not-show

# 3. 可视化配置（包含Albumentations）
echo "📊 3. 生成可视化配置（包含Albumentations）..."
python tools/visualization/browse_dataset.py works/coat_cls/coat_color_classification_show.py \
    --phase train \
    --mode pipeline \
    --show-number 5 \
    --output-dir visualization_results/show_config \
    --channel-order BGR \
    --not-show

echo "✅ 所有可视化完成！"
echo "📁 结果保存在: visualization_results/"
echo ""
echo "📋 生成的文件:"
echo "  - baseline/     : 基础配置（原始增强）"
echo "  - enhanced/     : 增强配置（综合增强）"
echo "  - show_config/  : 可视化配置（包含Albumentations）"
echo ""
echo "💡 您可以使用以下命令查看结果:"
echo "  - ls visualization_results/"
echo "  - 打开生成的PNG文件查看增强效果"
