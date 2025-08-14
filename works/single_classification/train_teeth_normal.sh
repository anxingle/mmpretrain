#!/bin/bash
# 齿痕舌与正常舌象二分类训练脚本

# 设置PYTHONPATH
export PYTHONPATH=/home/an/mmpretrain:$PYTHONPATH

# 创建工作目录
mkdir -p ./logs

echo "=========================================="
echo "开始训练: 齿痕舌(Teeth) vs 正常(Normal) 分类"
echo "=========================================="

# 正确的训练命令
cd /home/an/mmpretrain
export CUDA_VISIBLE_DEVICES=0 && python tools/train.py \
    works/single_classification/teeth_normal_classification.py \
    --work-dir ./logs/teeth_cls \
    2>&1 | tee ./training_teeth.log

echo "训练完成！"
