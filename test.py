import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms as T
import argparse
from runpy import run_path
from tqdm import tqdm
from PIL import Image

# ----------------------------
# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./checkpoints/derain_best_best.pth', help='模型checkpoint路径')
parser.add_argument('--save_dir', type=str, default='./results_real', help='保存去雨图像的文件夹')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512  # 只是占位，测试时保持原图大小
os.makedirs(args.save_dir, exist_ok=True)

# ----------------------------
# 加载 Restormer 模型
parameters = {
    'inp_channels': 3, 'out_channels': 3, 'dim': 48,
    'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
    'bias': False, 'LayerNorm_type': 'WithBias',
    'dual_pixel_task': False
}

load_arch = run_path(os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters).to(DEVICE)

# 加载 checkpoint
print(f"[√] 加载模型: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ----------------------------
# 测试集
class UnlabeledTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        original_size = img.size  # (w, h)
        return self.to_tensor(img), path, original_size
# 设置测试路径（根据需要修改）
TEST_DIR = "./Dataset/Real/test/input"
test_dataset = UnlabeledTestDataset(TEST_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# ----------------------------
# 推理 & 保存结果
with torch.no_grad():
    for img, path, original_size in tqdm(test_loader, desc="Testing"):
        img = img.to(DEVICE)
        _, _, h, w = img.shape

        # --------- Padding to make size divisible by 8 ---------
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

        # 推理
        output = model(img_padded).clamp(0, 1)

        # 裁剪回原图大小
        output = output[:, :, :h, :w]

        # 保存图像
        fname = os.path.basename(path[0])
        save_path = os.path.join(args.save_dir, fname)

        out_img = T.ToPILImage()(output.squeeze(0).cpu())
        out_img.save(save_path)

print(f"[√] 所有图像保存至: {args.save_dir}")
