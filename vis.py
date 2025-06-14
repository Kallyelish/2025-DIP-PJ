import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os

def extract_high_freq_component(x_fft_mag, radius=0.1):
    """
    提取高频区域：默认保留图像频率半径 > 0.1 的部分
    """
    B, C, H, W = x_fft_mag.shape
    cy, cx = H // 2, W // 2
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = ((X - cx) ** 2 + (Y - cy) ** 2).sqrt().to(x_fft_mag.device)
    mask = (dist > radius * H).float()
    return x_fft_mag * mask

def save_fft_and_highfreq_images_rgb(output, save_prefix="fft_rgb", radius=0.1):
    """
    处理 RGB 图像的 FFT 和高频提取并保存图像
    """
    if output.dim() != 4 or output.shape[1] != 3:
        raise ValueError("Expected RGB image with shape [B, 3, H, W]")

    B, C, H, W = output.shape
    for b in range(B):
        fft_vis = []
        high_vis = []
        for c in range(C):
            image = output[b, c]  # (H, W)
            image = image.float()

            # 计算 FFT 并中心化
            fft_result = torch.fft.fft2(image, norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_result)
            fft_mag = torch.abs(fft_shifted).unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)

            # 取 log 显示
            fft_log = torch.log1p(fft_mag)
            fft_vis.append(fft_log[0, 0].cpu().numpy())

            # 高频部分
            high_freq_mag = extract_high_freq_component(fft_mag, radius=radius)
            high_freq_log = torch.log1p(high_freq_mag)
            high_vis.append(high_freq_log[0, 0].cpu().numpy())

        # 合并为 RGB 幅度图
        fft_rgb = np.stack(fft_vis, axis=-1)
        high_rgb = np.stack(high_vis, axis=-1)

        # 保存
        plt.imsave(f"{save_prefix}_b{b}_fft.png", fft_rgb / fft_rgb.max())
        plt.imsave(f"{save_prefix}_b{b}_highfreq.png", high_rgb / high_rgb.max())

def load_image_as_tensor(image_path, size=None):
    """
    加载图像并转换为 4D Tensor (1, 3, H, W)，像素值归一化到 [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size) if size else transforms.Lambda(lambda x: x),
        transforms.ToTensor()  # [0,1], shape: (3, H, W)
    ])
    img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, H, W)
    return img_tensor

# ========== 示例 ==========
if __name__ == "__main__":
    image_path = "./Dataset/RE-RAIN/61.png"  # 替换为你的图片路径
    output = load_image_as_tensor(image_path, size=(800, 501))  # 可选 resize

    save_prefix = os.path.splitext(os.path.basename(image_path))[0]
    save_fft_and_highfreq_images_rgb(output, save_prefix=save_prefix, radius=0.1)

    print(f"保存完成：{save_prefix}_b0_fft.png 和 {save_prefix}_b0_highfreq.png")
