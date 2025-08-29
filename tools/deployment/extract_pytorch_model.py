import torch
import torch.nn as nn
from mmpretrain import init_model
from torchvision.models import efficientnet_v2_s

def extract_pure_pytorch_model(config_path, checkpoint_path, output_path, img_size: tuple):
    """提取为纯 PyTorch 模型，不依赖 mmpretrain"""
    
    # 加载 mmpretrain 模型
    mmp_model = init_model(config_path, checkpoint_path, device='cpu')
    mmp_model.eval()
    img = torch.rand(1, 3, img_size[0], img_size[1])
    traced = torch.jit.trace(mmp_model, img)  # 假设输入图像大小为 img_size x img_size
    traced.save(output_path)

    return output_path

if __name__ == '__main__':
    config_path = '/home/an/mmpretrain/works/coat_cls/coat_color_classification.py'
    checkpoint_path = '/data-ssd/logs/coat_color_cls_v3/epoch_400.pth'
    output_path = '/data-ssd/logs/coat_color_cls_v3/coat_color_efficientNetV2.pt'
    
    extract_pure_pytorch_model(config_path, checkpoint_path, output_path, (384, 384))