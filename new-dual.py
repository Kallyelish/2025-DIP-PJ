import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os, argparse, time
from PIL import Image
from tqdm import tqdm
from runpy import run_path
from skimage.metrics import structural_similarity as ssim

# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 8
LR = 2e-4
IMG_SIZE = 64
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==================== Frequency Encoder Branch =====================
class FrequencyEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=128):
        super(FrequencyEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.encoder(x)
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
# ----------------------------
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1.0 / mse) if mse != 0 else float('inf')

class PairedImageDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        self.mode = mode
        self.rain_dir = os.path.join(data_path, 'input')
        self.clean_dir = os.path.join(data_path, 'gt')
        rain_images = sorted([os.path.join(self.rain_dir, x) for x in os.listdir(self.rain_dir)])
        clean_images = sorted([os.path.join(self.clean_dir, x) for x in os.listdir(self.clean_dir)])
        if mode == 'train':
            max_samples = 2000
            indices = torch.randperm(len(rain_images))[:max_samples]
            self.rain_images = [rain_images[i] for i in indices]
            self.clean_images = [clean_images[i] for i in indices]
        else:
            max_samples = 50
            self.rain_images = rain_images[:max_samples]
            self.clean_images = clean_images[:max_samples]
        self.resize_transform = T.Resize((IMG_SIZE, IMG_SIZE))
        self.to_tensor = T.ToTensor()
    def __len__(self):
        return len(self.rain_images)

    def __getitem__(self, idx):
        rain_path = self.rain_images[idx]
        clean_path = self.clean_images[idx]

        rain_img = Image.open(rain_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        if self.mode == 'train':
            rain_img = self.resize_transform(rain_img)
            clean_img = self.resize_transform(clean_img)

        rain_img = self.to_tensor(rain_img)
        clean_img = self.to_tensor(clean_img)

        return rain_img, clean_img

# ----------------------------
class SSIMLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.clamp(0, 1).cpu().detach().numpy().transpose(0, 2, 3, 1)
        target = target.clamp(0, 1).cpu().detach().numpy().transpose(0, 2, 3, 1)
        ssim_val = 0
        for i in range(pred.shape[0]):
            ssim_val += ssim(pred[i], target[i], channel_axis=2, data_range=1.0)
        return 1 - ssim_val / pred.shape[0]

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features.eval().to(DEVICE)
        self.feature_extractor = create_feature_extractor(vgg, return_nodes={"16": "feat"}).to(DEVICE)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_feat = self.feature_extractor(pred)['feat']
        target_feat = self.feature_extractor(target)['feat']
        return F.l1_loss(pred_feat, target_feat)

class MultiLevelContrastLoss(nn.Module):
    def __init__(self, temperature=0.1, margin=0.3):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Triplet loss
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        triplet = F.relu(pos_dist - neg_dist + self.margin).mean()

        # InfoNCE
        anchor_n = F.normalize(anchor, dim=1)
        positive_n = F.normalize(positive, dim=1)
        logits = torch.mm(anchor_n, positive_n.T) / self.temperature
        labels = torch.arange(len(anchor)).to(anchor.device)
        info_nce = F.cross_entropy(logits, labels)

        return 0.1 * triplet + info_nce

# ----------------------------
def get_weights_and_parameters(parameters):
    weights = os.path.join('deraining.pth')
    return weights, parameters

parameters = {
    'inp_channels': 3, 'out_channels': 3, 'dim': 48,
    'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
    'bias': False, 'LayerNorm_type': 'WithBias',
    'dual_pixel_task': False
}

weights, parameters = get_weights_and_parameters(parameters)
load_arch = run_path(os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters).to(DEVICE)
freq_encoder = FrequencyEncoder().to(DEVICE)
optimizer = optim.Adam(
    list(model.parameters()) + list(freq_encoder.parameters()), lr=LR
)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
scaler = GradScaler()

l1_loss = nn.L1Loss()
ssim_loss = SSIMLoss()
percep_loss = PerceptualLoss()
contrast_loss = MultiLevelContrastLoss(temperature=0.1, margin=0.3)

start_epoch = 0
if args.resume:
    print(f"==> 恢复训练: {args.resume}")
    ckpt = torch.load(args.resume, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
else:
    checkpoint = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(checkpoint['params'])
    print("==> 载入预训练 deraining.pth")

# ----------------------------
train_loader = DataLoader(PairedImageDataset("./Dataset/Real/train"), batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
val_loader = DataLoader(PairedImageDataset("./Dataset/Real/test", mode='val'), batch_size=1, num_workers=4, shuffle=False)

best_psnr = 0
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

    for rain, clean in loop:
        rain, clean = rain.to(DEVICE), clean.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            output = model(rain)

            # L1, SSIM, perceptual
            l1_v = l1_loss(output, clean)
            ssim_v = ssim_loss(output, clean)
            percep_v = percep_loss(output, clean)

            # FFT domain
            fft_output = torch.fft.fft2(output, norm='ortho')
            fft_clean = torch.fft.fft2(clean, norm='ortho')
            fft_input = torch.fft.fft2(rain, norm='ortho')

            fft_output_amp = torch.abs(fft_output)
            fft_clean_amp = torch.abs(fft_clean)
            fft_input_amp = torch.abs(fft_input)

            # 高频提取
            hf_output = extract_high_freq_component(fft_output_amp)  # anchor
            hf_clean = extract_high_freq_component(fft_clean_amp)    # positive
            hf_input = extract_high_freq_component(fft_input_amp)    # negative

            # 特征提取
            feat_output = freq_encoder(hf_output)
            feat_clean = freq_encoder(hf_clean)
            feat_input = freq_encoder(hf_input)

            # 多级对比损失
            contrast_v = contrast_loss(feat_output, feat_clean, feat_input)

            # FFT loss
            fft_v = F.l1_loss(fft_output_amp, fft_clean_amp)

            # 加权损失组合
            w_l1, w_ssim, w_percep, w_contrast, w_fft = 1.0, 0.5, 0.1, 0.1, 0.05
            # w_l1, w_percep= 1.0, 1.0
            loss = (
                w_l1 * l1_v +
                w_ssim * ssim_v +
                w_percep * percep_v +
                w_contrast * contrast_v +
                w_fft * fft_v
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n + 1))

    model.eval()
    val_psnr = 0
    with torch.no_grad():
        for val_rain, val_clean in tqdm(val_loader, desc="Validating"):
            val_rain, val_clean = val_rain.to(DEVICE), val_clean.to(DEVICE)
            val_out = model(val_rain)
            val_psnr += calculate_psnr(val_out.clamp(0, 1), val_clean)
    avg_val_psnr = val_psnr / len(val_loader)
    print(f"Validation PSNR: {avg_val_psnr:.2f} dB")
    scheduler.step(avg_val_psnr)

    # 保存最好模型
    if avg_val_psnr > best_psnr:
        best_psnr = avg_val_psnr
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'freq_encoder_state_dict': freq_encoder.state_dict(),
            # 'dynamic_weights_state_dict': dynamic_weights.state_dict(),
            'psnr': best_psnr,
        }, os.path.join(CHECKPOINT_DIR, 'derain_best.pth'))
        print(f"[√] 最佳模型已保存，PSNR: {best_psnr:.2f} dB")
