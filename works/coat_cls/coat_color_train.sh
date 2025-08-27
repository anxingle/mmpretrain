#!/bin/bash
# 齿痕舌与正常舌象二分类训练脚本

# 设置PYTHONPATH
export PYTHONPATH=/home/an/mmpretrain:$PYTHONPATH

# 创建工作目录
mkdir -p ./logs

echo "=========================================="
echo "开始训练: 舌苔颜色 灰(gray)/白(white)/黄(yellow) 分类"
echo "=========================================="

# 正确的训练命令
cd /home/an/mmpretrain
export CUDA_VISIBLE_DEVICES=0 && python tools/train.py \
    works/coat_cls/coat_color_classification.py \
    --work-dir /data-ssd/logs/coat_color_cls \
    2>&1 | tee ./training_coat_color.log

echo "训练完成！"
