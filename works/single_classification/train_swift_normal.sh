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
    --work-dir ./logs/swift_cls \
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


08/12 16:11:30 - mmengine - INFO - Exp name: crack_normal_classification_20250812_154010
08/12 16:11:30 - mmengine - INFO - Saving checkpoint at 5 epochs
08/12 16:13:14 - mmengine - INFO - Epoch(val) [5][344/344]    accuracy/top1: 88.8626  single-label/precision_classwise: [85.03936767578125, 91.10022735595703]  single-label/recall_classwise: [84.83063507080078, 91.23161315917969]  single-label/f1-score_classwise: [84.93487548828125, 91.1658706665039]  data_time: 0.0629  time: 0.2944
08/12 16:13:16 - mmengine - INFO - The best checkpoint with 88.8626 accuracy/top1 at 5 epoch is saved to best_accuracy_top1_epoch_5.pth.
08/12 16:13:47 - mmengine - INFO - Epoch(train)   [6][ 50/688]  base_lr: 5.0769e-04 lr: 5.0769e-04  eta: 2 days, 13:50:02  time: 0.5462  data_time: 0.0044  memory: 20391  grad_norm: 3.1294  loss: 0.2694
08/12 16:14:15 - mmengine - INFO - Epoch(train)   [6][100/688]  base_lr: 5.1495e-04 lr: 5.1495e-04  eta: 2 days, 13:50:04  time: 0.5469  data_time: 0.0043  memory: 20391  grad_norm: 4.0522  loss: 0.3746