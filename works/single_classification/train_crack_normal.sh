#!/bin/bash
# 裂纹与正常舌象二分类训练脚本

# 设置PYTHONPATH
export PYTHONPATH=/home/an/mmpretrain:$PYTHONPATH

# 创建工作目录
mkdir -p ../crack_normal_classification

echo "=========================================="
echo "开始训练: 裂纹(Crack) vs 正常(Normal) 分类"
echo "数据集: crack_VS_normal_0812"
echo "=========================================="

# 单GPU训练
python ../../tools/train.py crack_normal_classification.py \
    --work-dir ../crack_normal_classification \
    2>&1 | tee ../crack_normal_classification/train.log

echo "训练完成！模型保存在: works/crack_normal_classification/"

# 多GPU训练（如果需要，取消下面的注释）
# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --master_port=29501 \
#     ../../tools/train.py \
#     crack_normal_classification.py \
#     --launcher pytorch \
#     --work-dir ../crack_normal_classification \
#     2>&1 | tee ../crack_normal_classification/train.log
