#!/bin/bash
# 点刺与正常舌象二分类训练脚本

# 设置PYTHONPATH
export PYTHONPATH=/home/an/mmpretrain:$PYTHONPATH

# 创建工作目录
mkdir -p ./logs

echo "=========================================="
echo "开始训练: 点刺(Swift) vs 正常(Normal) 分类"
echo "数据集: swift_VS_normal_0812"
echo "=========================================="

# 单GPU训练
cd /home/an/mmpretrain
export CUDA_VISIBLE_DEVICES=1 && python tools/train.py \
    works/single_classification/swift_normal_classification.py \
    --work-dir /data/logs/swift_cls \
    2>&1 | tee ./training_swift.log

echo "训练完成！模型保存在: works/single_classification/logs/swift_cls"

# 多GPU训练（如果需要，取消下面的注释）
# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --master_port=29502 \
#     ../../tools/train.py \
#     swift_normal_classification.py \
#     --launcher pytorch \
#     --work-dir ../swift_normal_classification \
#     2>&1 | tee ../swift_normal_classification/train.log

