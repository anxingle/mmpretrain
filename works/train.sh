# 可视化数据增强的脚本
# python ./tools/visualization/browse_dataset.py ./works/efficientNetV2_b0_v0.py --phase train --output-dir tmp2 --mode pipeline --show-number 100 --rescale-factor 1 --channel-order BGR

# 训练脚本 EfficientNetV2_b0_v0
# python ./tools/train.py ./works/efficientNetV2_b0_v0.py --work-dir $PWD/works/efficientNetV2_b0_v0 --auto-scale-lr 

# 训练脚本 EfficientNetV2_xl_v0
python ./tools/train.py ./works/efficientNetV2_xl_v0.py --work-dir $PWD/works/efficientNetV2_xl_v0 --auto-scale-lr

python ./tools/train.py ./works/efficientNetV2_b0_v2.py --work-dir $PWD/works/train_interval_log/efficientNetV2_b0_v2_color --auto-scale-lr