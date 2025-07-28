# 执行测试
python tools/test.py works/efficientNetV2_b0_v2.py works/train_interval_log/efficientNetV2_b0_v2_color/epoch_1000.pth --work-dir works/train_interval_log/efficientNetV2_b0_v2_color --out b0_v2_color.pkl --show-dir works/train_interval_log/efficientNetV2_b0_v2_color

# 计算出各个类别的混淆矩阵
python tools/analysis_tools/confusion_matrix.py  works/efficientNetV2_b0_v2.py works/train_interval_log/efficientNetV2_b0_v2_color/b0_v2_color.pkl --show-path  works/train_interval_log/efficientNetV2_b0_v2_color/confusion_result.png  --include-values  --cmap Reds


python tools/test.py works/efficientNetV2_xl_v2.py works/efficientNetV2_xl_v2_color/epoch_535.pth --work-dir works/train_interval_log/efficientNetV2_xl_v2_color --out works/train_interval_log/efficientNetV2_xl_v2_color/xl_v2_color.pkl --show-dir works/train_interval_log/efficientNetV2_xl_v2_color
# 计算出各个类别的混淆矩阵
python tools/analysis_tools/confusion_matrix.py  works/efficientNetV2_xl_v2.py works/train_interval_log/efficientNetV2_xl_v2_color/xl_v2_color.pkl --show-path  works/train_interval_log/efficientNetV2_xl_v2_color/confusion_result.png  --include-values  --cmap Reds