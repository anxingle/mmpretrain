# 查看经过数据增强处理后的图片
python tools/visualization/browse_dataset.py works/single_classification/swift_normal_classification.py \
    --phase train \
    --mode transformed \
    --show-number 20 \
    --output-dir visualize_augmented \
    --channel-order RGB \
    --not-show

# 查看原图与增强后图片的对比
python tools/visualization/browse_dataset.py works/single_classification/swift_normal_classification.py \
    --phase train \
    --mode concat \
    --show-number 20 \
    --output-dir visualize_compare \
    --channel-order RGB \
    --not-show

# 查看数据处理流水线的每个步骤
python tools/visualization/browse_dataset.py works/single_classification/swift_normal_classification.py \
    --phase train \
    --mode pipeline \
    --show-number 10 \
    --output-dir visualize_pipeline \
    --channel-order RGB \
    --not-show