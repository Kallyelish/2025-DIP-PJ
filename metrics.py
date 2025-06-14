import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS

# 初始化LPIPS模型（使用AlexNet版本）
lpips_model = LPIPS(net='alex').cuda() if torch.cuda.is_available() else LPIPS(net='alex')

def load_image_pair(pred_path, gt_path):
    """加载预测图像和GT图像"""
    pred_img = Image.open(pred_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('RGB')
    return ToTensor()(pred_img), ToTensor()(gt_img)

def calculate_psnr(pred, gt, max_val=1.0):
    """计算PSNR"""
    mse = torch.mean((pred - gt) ** 2)
    return 10 * torch.log10(max_val**2 / mse)

def calculate_ssim(pred, gt):
    """计算SSIM（使用skimage实现）"""
    pred_np = pred.permute(1, 2, 0).numpy()
    gt_np = gt.permute(1, 2, 0).numpy()
    return ssim(pred_np, gt_np, channel_axis=2, data_range=1.0)

def calculate_lpips(pred, gt):
    """计算LPIPS（需要图像范围为[-1,1]）"""
    pred = pred.unsqueeze(0).cuda() * 2 - 1  # [0,1] -> [-1,1]
    gt = gt.unsqueeze(0).cuda() * 2 - 1
    return lpips_model(pred, gt).item()

def evaluate_folders(pred_dir, gt_dir):
    """主评估函数"""
    psnr_list, ssim_list, lpips_list = [], [], []
    filenames = sorted(os.listdir(gt_dir))
    
    for filename in tqdm(filenames, desc="Evaluating"):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        
        if not os.path.exists(pred_path):
            print(f"Warning: {filename} not found in pred directory")
            continue
            
        pred, gt = load_image_pair(pred_path, gt_path)
        
        # 计算指标
        psnr = calculate_psnr(pred, gt)
        ssim_val = calculate_ssim(pred, gt)
        lpips_val = calculate_lpips(pred, gt)
        
        psnr_list.append(psnr.item())
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
    
    # 统计结果
    metrics = {
        'PSNR': {'mean': np.mean(psnr_list), 'std': np.std(psnr_list)},
        'SSIM': {'mean': np.mean(ssim_list), 'std': np.std(ssim_list)},
        'LPIPS': {'mean': np.mean(lpips_list), 'std': np.std(lpips_list)}
    }
    
    # 打印结果
    print("\nEvaluation Results:")
    print(f"{'Metric':<10} | {'Mean':<8} | {'Std':<6}")
    print("-"*30)
    for name, values in metrics.items():
        print(f"{name:<10} | {values['mean']:.4f} | {values['std']:.4f}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='预测结果文件夹路径')
    parser.add_argument('--gt', type=str, required=True, help='真实GT文件夹路径')
    args = parser.parse_args()
    
    # 检查文件夹存在
    assert os.path.isdir(args.pred), f"预测文件夹不存在: {args.pred}"
    assert os.path.isdir(args.gt), f"GT文件夹不存在: {args.gt}"
    
    # 运行评估
    results = evaluate_folders(args.pred, args.gt)