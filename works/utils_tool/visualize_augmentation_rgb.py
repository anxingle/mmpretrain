# save as: visualize_augmentation_fixed.py
import os
import sys
import cv2
import numpy as np
import random

# 确保可以正确导入mmpretrain
mmpretrain_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, mmpretrain_root)
print(f"MMPretrain root: {mmpretrain_root}")

from mmengine.config import Config
from mmcv.transforms import Compose

# 导入mmpretrain模块
try:
    import mmpretrain
    from mmpretrain.datasets.builder import build_dataset
    # 导入所有变换模块以确保注册
    import mmpretrain.datasets.transforms.processing
    import mmpretrain.datasets.transforms.auto_augment
    import mmpretrain.datasets.transforms.formatting
    # 从mmcv导入基础变换
    from mmcv.transforms import Pad, LoadImageFromFile, RandomFlip, CenterCrop
    # 从mmpretrain导入特定变换
    from mmpretrain.datasets.transforms import ResizeEdge, PackInputs, ColorJitter, GaussianBlur
    from mmpretrain.registry import TRANSFORMS
    print("成功导入 mmpretrain 模块😄")
except ImportError as e:
    print(f"导入 mmpretrain 失败: {e}")
    sys.exit(1)

def save_augmentation_samples_rgb(config_file, num_samples=10, save_dir='augmentation_samples_rgb'):
    """
    数据增强可视化，保存为正确的RGB格式
    """

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'augmented'), exist_ok=True)

    print("正在加载配置...")

    # 转换为绝对路径
    if not os.path.isabs(config_file):
        config_file = os.path.join(mmpretrain_root, config_file)

    print(f"配置文件路径: {config_file}")

    # 加载配置
    cfg = Config.fromfile(config_file)

    # 构建数据集
    dataset = build_dataset(cfg.train_dataloader.dataset)
    print(f"数据集大小: {len(dataset)}")

    # 创建原始数据pipeline（无增强）
    original_pipeline = Compose([
        LoadImageFromFile(),
        ResizeEdge(scale=384, edge='long', backend='cv2'),
        Pad(size=(384, 384), pad_val=0, padding_mode='constant'),
    ])

    # 创建增强数据pipeline（从配置中获取，去除PackInputs）
    augmented_pipeline_cfg = cfg.train_pipeline[:-1]  # 除了PackInputs
    print("增强pipeline配置:")
    for i, step in enumerate(augmented_pipeline_cfg):
        print(f"  {i+1}. {step}")

    # 手动构建增强管道
    augmented_transforms = []
    for transform_cfg in augmented_pipeline_cfg:
        if transform_cfg['type'] == 'LoadImageFromFile':
            augmented_transforms.append(LoadImageFromFile())
        elif transform_cfg['type'] == 'ResizeEdge':
            augmented_transforms.append(ResizeEdge(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'RandomFlip':
            augmented_transforms.append(RandomFlip(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'CenterCrop':
            augmented_transforms.append(CenterCrop(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'Pad':
            augmented_transforms.append(Pad(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'ColorJitter':
            augmented_transforms.append(ColorJitter(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'GaussianBlur':
            augmented_transforms.append(GaussianBlur(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        # 可以根据需要添加更多变换类型
        else:
            print(f"警告: 未处理的变换类型 {transform_cfg['type']}")
    
    augmented_pipeline = Compose(augmented_transforms)

    # 随机选择样本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    print(f"开始处理 {len(indices)} 个样本...")

    for i, idx in enumerate(indices):
        # 获取原始数据信息
        data_info = dataset.get_data_info(idx)
        img_name = os.path.basename(data_info["img_path"]).split('.')[0]

        print(f"处理样本 {i+1}/{len(indices)}: {img_name}")
        print(f"  图片路径: {data_info['img_path']}")

        # 处理原始图片
        try:
            original_data = original_pipeline(data_info.copy())
            original_img = original_data['img']

            # 转换为RGB格式并保存
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_save_path = os.path.join(save_dir, 'original', f'{img_name}_original.jpg')
            cv2.imwrite(original_save_path, cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR))

            # 生成多个增强版本
            for j in range(5):  # 生成5个增强版本
                augmented_data = augmented_pipeline(data_info.copy())
                augmented_img = augmented_data['img']

                # 转换为RGB格式并保存
                augmented_img_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
                augmented_save_path = os.path.join(save_dir, 'augmented', f'{img_name}_aug_{j+1}.jpg')
                cv2.imwrite(augmented_save_path, cv2.cvtColor(augmented_img_rgb, cv2.COLOR_RGB2BGR))

            print(f"  ✓ 已保存原图: {original_save_path}")
            print(f"  ✓ 已保存5个增强版本")

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n完成！结果保存在: {save_dir}")
    print(f"- 原图目录: {os.path.join(save_dir, 'original')}")
    print(f"- 增强图目录: {os.path.join(save_dir, 'augmented')}")

def create_side_by_side_comparison_rgb(config_file, save_dir='comparison_rgb', num_samples=5):
    """
    创建原图与增强图的并排对比，确保RGB格式正确
    """

    os.makedirs(save_dir, exist_ok=True)

    # 转换为绝对路径
    if not os.path.isabs(config_file):
        config_file = os.path.join(mmpretrain_root, config_file)

    # 加载配置
    cfg = Config.fromfile(config_file)
    dataset = build_dataset(cfg.train_dataloader.dataset)

    # 创建pipeline
    original_pipeline = Compose([
        LoadImageFromFile(),
        ResizeEdge(scale=384, edge='long', backend='cv2'),
        Pad(size=(384, 384), pad_val=0, padding_mode='constant'),
    ])

    # 手动构建增强管道
    augmented_pipeline_cfg = cfg.train_pipeline[:-1]
    augmented_transforms = []
    for transform_cfg in augmented_pipeline_cfg:
        if transform_cfg['type'] == 'LoadImageFromFile':
            augmented_transforms.append(LoadImageFromFile())
        elif transform_cfg['type'] == 'ResizeEdge':
            augmented_transforms.append(ResizeEdge(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'RandomFlip':
            augmented_transforms.append(RandomFlip(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'CenterCrop':
            augmented_transforms.append(CenterCrop(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'Pad':
            augmented_transforms.append(Pad(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'ColorJitter':
            augmented_transforms.append(ColorJitter(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        elif transform_cfg['type'] == 'GaussianBlur':
            augmented_transforms.append(GaussianBlur(**{k: v for k, v in transform_cfg.items() if k != 'type'}))
        else:
            print(f"警告: 未处理的变换类型 {transform_cfg['type']}")
    
    augmented_pipeline = Compose(augmented_transforms)

    # 随机选择样本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        data_info = dataset.get_data_info(idx)
        img_name = os.path.basename(data_info["img_path"]).split('.')[0]

        print(f"创建对比图 {i+1}/{len(indices)}: {img_name}")

        try:
            # 处理图片
            original_data = original_pipeline(data_info.copy())
            original_img = original_data['img']

            # 转换为RGB
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # 创建对比图
            comparison_images = [original_img_rgb]

            # 生成5个增强版本
            for j in range(5):
                augmented_data = augmented_pipeline(data_info.copy())
                augmented_img = augmented_data['img']
                # 转换为RGB
                augmented_img_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
                comparison_images.append(augmented_img_rgb)

            # 将图片水平拼接
            concatenated = np.hstack(comparison_images)

            # 添加文字标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (0, 0, 0)  # RGB格式下的黑色
            thickness = 2

            labels = ['Original'] + [f'Aug{j+1}' for j in range(5)]
            for k, label in enumerate(labels):
                x_pos = k * 384 + 10
                cv2.putText(concatenated, label, (x_pos, 30), font, font_scale, color, thickness)

            # 保存对比图（转换回BGR用于保存）
            save_path = os.path.join(save_dir, f'{img_name}_comparison_rgb.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(concatenated, cv2.COLOR_RGB2BGR))

            print(f"  ✓ 保存对比图: {save_path}")

        except Exception as e:
            print(f"  ✗ 创建对比图失败: {e}")

if __name__ == '__main__':
    # 配置文件相对于mmpretrain根目录的路径
    config_file = 'works/single_classification/swift_normal_classification.py'

    print("="*60)
    print("开始RGB格式数据增强可视化...")
    print("="*60)

    try:
        # 检查当前工作目录
        print(f"当前工作目录: {os.getcwd()}")
        print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")

        # 方法1: 批量处理多个样本
        save_augmentation_samples_rgb(config_file, num_samples=10, save_dir='augmentation_samples_rgb')

        print("\n" + "="*50)

        # 方法2: 创建对比图
        create_side_by_side_comparison_rgb(config_file, save_dir='comparison_rgb', num_samples=5)

        print("="*60)
        print("RGB格式可视化完成!")
        print("="*60)

    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("1. 是否在正确的环境中运行 (openmmlab)")
        print("2. mmpretrain 是否正确安装")
        print("3. 配置文件路径是否正确")
        print("4. 数据集路径是否存在")