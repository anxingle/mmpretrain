# MMPretrain UV 使用指南

## 概述

现在您可以使用 `uv` 来管理 mmpretrain 项目！🎉

## 基本命令

### 激活环境并运行代码
```bash
# 直接在 uv 环境中运行 Python
uv run python your_script.py

# 或者激活虚拟环境
source .venv/bin/activate
python your_script.py
```

### 依赖管理
```bash
# 同步所有依赖
uv sync

# 添加新依赖
uv add package_name

# 添加开发依赖
uv add --dev package_name

# 移除依赖
uv remove package_name

# 查看依赖树
uv tree
```

### 安装可选依赖组合
```bash
# 安装多模态相关依赖
uv sync --extra multimodal

# 安装测试相关依赖
uv sync --extra tests

# 安装所有依赖
uv sync --extra all
```

## 项目结构

- `pyproject.toml` - 主要配置文件，包含所有依赖定义
- `.venv/` - 虚拟环境目录（由 uv 自动管理）
- `uv.lock` - 锁定文件，确保依赖版本一致性

## 使用示例

### 基本图像分类训练
```bash
# 使用 uv 运行训练脚本
uv run python tools/train.py configs/resnet/resnet50_8xb32_in1k.py

# 或者激活环境后运行
source .venv/bin/activate
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py
```

### 图像推理
```bash
uv run python demo/image_demo.py demo/demo.JPEG configs/resnet/resnet50_8xb32_in1k.py --device cpu
```

## 当前已安装的依赖组

✅ **核心依赖 (runtime)**:
- einops, matplotlib, numpy, rich 等

✅ **OpenMMLab 生态**:
- mmcv (>= 2.0.0, < 2.4.0)
- mmengine (>= 0.8.3, < 1.0.0)

✅ **深度学习框架**:
- torch (>= 2.8.0，支持 CUDA 12.8，兼容大多数现代 GPU)
- torchvision (>= 0.23.0)

✅ **所有可选依赖 (all)**:
- 图像增强: albumentations
- 可视化: grad-cam
- 网络请求: requests
- 机器学习: scikit-learn
- 测试工具: pytest, coverage
- 多模态: transformers, pycocotools

## 优势

1. **更快的依赖解析**: uv 比 pip 快 10-100 倍
2. **更好的依赖管理**: 自动解决依赖冲突
3. **锁定文件**: 确保团队间环境一致性
4. **现代化工具**: 支持 PEP 621 标准

## CUDA 12.9+ 支持

当前项目使用 PyTorch 2.8.0 (CUDA 12.8)，对于需要 CUDA 12.9+ 支持的用户，有以下选项：

### 选项 1: 使用 PyTorch Nightly 版本
```bash
# 取消注释 pyproject.toml 中的 nightly 配置，然后运行:
uv sync
```

### 选项 2: 等待官方稳定版本
PyTorch 团队正在开发对 CUDA 12.9+ 的官方支持，建议关注 [PyTorch 发布页面](https://pytorch.org/)。

### 选项 3: 从源代码构建
高级用户可以从 PyTorch 源代码构建支持最新 CUDA 的版本。

## 注意事项

- 虚拟环境位于 `.venv/` 目录，请不要手动修改
- 如需要添加新依赖，使用 `uv add` 而不是 `pip install`
- 项目的 `requirements/*.txt` 文件仍然保留，但现在由 `pyproject.toml` 统一管理
- 项目现在需要 Python 3.9+ (为了支持最新的 PyTorch 版本)

## 验证安装

运行以下命令验证一切正常：
```bash
uv run python -c "import mmpretrain; print(f'MMPretrain version: {mmpretrain.__version__}')"
```

应该输出: `MMPretrain version: 1.2.0`
