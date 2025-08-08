#!/bin/bash
# 舌体形态多标签分类 - 快速开始脚本
# 自动完成数据集转换和模型训练

set -e  # 遇到错误立即退出

echo "🚀 舌体形态多标签分类 - 快速开始"
echo "=========================================="

# 检查当前目录
if [[ ! -f "convert_body_dataset.py" ]]; then
    echo "❌ 请在 /home/an/mmpretrain/works/multi_label 目录下运行此脚本"
    exit 1
fi

# 步骤1: 转换数据集
echo "📊 步骤1: 转换数据集格式"
echo "------------------------------------------"
python convert_body_dataset.py

if [[ $? -ne 0 ]]; then
    echo "❌ 数据集转换失败"
    exit 1
fi

echo "✅ 数据集转换完成"
echo

# 步骤2: 检查转换结果
echo "📋 步骤2: 检查转换结果"
echo "------------------------------------------"

# 检查训练集
if [[ -f "../datasets/body3_multilabel_train/statistics_report.json" ]]; then
    echo "📊 训练集统计:"
    python -c "
import json
with open('../datasets/body3_multilabel_train/statistics_report.json', 'r') as f:
    stats = json.load(f)
print(f'  总样本数: {stats.get(\"total_samples\", \"未知\")}')
print(f'  训练集: {stats.get(\"train_samples\", \"未知\")}')
print(f'  验证集: {stats.get(\"val_samples\", \"未知\")}')
if 'class_distribution' in stats:
    print('  类别分布:')
    for cls, count in stats['class_distribution'].items():
        print(f'    {cls}: {count}')
"
else
    echo "⚠️  训练集统计文件不存在"
fi

echo

# 检查测试集
if [[ -f "../datasets/body3_multilabel_test/statistics_report.json" ]]; then
    echo "📊 测试集统计:"
    python -c "
import json
with open('../datasets/body3_multilabel_test/statistics_report.json', 'r') as f:
    stats = json.load(f)
print(f'  总样本数: {stats.get(\"total_samples\", \"未知\")}')
if 'class_distribution' in stats:
    print('  类别分布:')
    for cls, count in stats['class_distribution'].items():
        print(f'    {cls}: {count}')
"
else
    echo "⚠️  测试集统计文件不存在"
fi

echo
echo "✅ 数据集检查完成"
echo

# 步骤3: 开始训练
echo "🏋️  步骤3: 开始模型训练"
echo "------------------------------------------"
echo "配置文件: efficientNetV2_xl_body_morphology_multilabel.py"
echo "模型: EfficientNetV2-XL"
echo "任务: 舌体形态多标签分类 (齿痕/点刺/裂纹/正常)"
echo

# 切换到mmpretrain根目录
cd /home/an/mmpretrain

# 检查配置文件
if [[ ! -f "works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py" ]]; then
    echo "❌ 配置文件不存在"
    exit 1
fi

echo "🚀 启动训练..."
echo "训练命令: python tools/train.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py"
echo

# 询问是否开始训练
read -p "是否立即开始训练? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🏃 开始训练..."
    python tools/train.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py
else
    echo "⏸️  训练已暂停"
    echo
    echo "📝 手动训练命令:"
    echo "   cd /home/an/mmpretrain"
    echo "   python tools/train.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py"
    echo
    echo "📝 测试命令:"
    echo "   python tools/test.py works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py \\"
    echo "       work_dirs/efficientNetV2_xl_body_morphology_multilabel/latest.pth"
fi

echo
echo "🎉 快速开始脚本执行完成！"
echo
echo "📁 生成的文件:"
echo "   数据集: works/datasets/body3_multilabel_train/"
echo "   数据集: works/datasets/body3_multilabel_test/"
echo "   配置: works/multi_label/efficientNetV2_xl_body_morphology_multilabel.py"
echo "   日志: work_dirs/efficientNetV2_xl_body_morphology_multilabel/"
echo
echo "📊 监控训练:"
echo "   tensorboard --logdir work_dirs/efficientNetV2_xl_body_morphology_multilabel/"
echo
echo "📖 更多信息请查看:"
echo "   works/multi_label/dataset_conversion_guide.md"
echo "   works/multi_label/multilabel_config_guide.md"