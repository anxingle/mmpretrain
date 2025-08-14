import torch
import torch.nn as nn
import os

def load_checkpoint_compatible(checkpoint_path):
    """Load checkpoint with PyTorch 2.6+ compatibility"""
    # 直接使用 weights_only=False 以保存完整的网络结构
    print("Loading checkpoint with weights_only=False to preserve network structure...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint

def extract_pure_pytorch_model(config_path, checkpoint_path, output_path, img_size: tuple):
    """提取为纯 PyTorch 模型，不依赖 mmpretrain"""
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 直接使用自定义方法加载模型，避免 init_model 的 weights_only 问题
    from mmpretrain.apis import get_model
    from mmengine.config import Config
    
    config = Config.fromfile(config_path)
    mmp_model = get_model(config, pretrained=False, device='cpu')
    
    # 使用我们的自定义函数加载权重
    checkpoint = load_checkpoint_compatible(checkpoint_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 移除 module. 前缀（如果有的话）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    mmp_model.load_state_dict(new_state_dict, strict=False)
    mmp_model.eval()
    print("Model loaded successfully with custom method!")
    
    # 创建测试输入
    img = torch.rand(1, 3, img_size[0], img_size[1])
    
    # 测试前向传播
    with torch.no_grad():
        try:
            output = mmp_model(img)
            print(f"Model forward pass successful! Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        except Exception as e:
            print(f"Warning: Forward pass failed: {e}")
            print("Continuing with tracing...")
    
    # 转换为 TorchScript
    try:
        traced = torch.jit.trace(mmp_model, img)
        traced.save(output_path)
        print(f"Model successfully traced and saved to: {output_path}")
    except Exception as e:
        print(f"Error during tracing: {e}")
        print("Trying to save the model directly...")
        
        # 如果 tracing 失败，尝试直接保存模型
        try:
            torch.save(mmp_model, output_path.replace('.pt', '_direct.pt'))
            print(f"Model saved directly to: {output_path.replace('.pt', '_direct.pt')}")
        except Exception as save_e:
            print(f"Error saving model: {save_e}")
            raise

    return output_path

if __name__ == '__main__':
    config_path = 'works/efficientNetV2_xl_v2_WCE.py'
    checkpoint_path = 'works/train_interval_log/efficientNetV2_xl_v2_WCE_color/epoch_400.pth'
    output_path = 'efficientNetV2_xl_v2_WCE.pt'
    
    extract_pure_pytorch_model(config_path, checkpoint_path, output_path, (384, 384))