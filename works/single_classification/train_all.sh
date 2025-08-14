#!/bin/bash
# 批量训练所有舌象分类模型

echo "==========================================="
echo "批量训练舌象分类模型"
echo "==========================================="
echo ""

# 训练齿痕舌分类
echo "[1/3] 训练齿痕舌(Teeth) vs 正常(Normal) 分类..."
bash train_teeth_normal.sh

# 训练裂纹分类
echo ""
echo "[2/3] 训练裂纹(Crack) vs 正常(Normal) 分类..."
bash train_crack_normal.sh

# 训练点刺分类
echo ""
echo "[3/3] 训练点刺(Swift) vs 正常(Normal) 分类..."
bash train_swift_normal.sh

echo ""
echo "==========================================="
echo "所有模型训练完成！"
echo "模型保存位置："
echo "  - works/teeth_normal_classification/"
echo "  - works/crack_normal_classification/"
echo "  - works/swift_normal_classification/"
echo "==========================================="
