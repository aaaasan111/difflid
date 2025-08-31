import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, utils
from PIL import Image
import os


# -----------------------------------------------------------------------------
# H-space 特征提取器：从预训练 DDPM 模型中获取不同时间步的 h-space 特征
# -----------------------------------------------------------------------------
class HSpaceExtractor(nn.Module):
    def __init__(self, ddpm_model, alpha_bar_t1, alpha_bar_t2, device='cuda'):
        super().__init__()
        self.model = ddpm_model.eval().to(device)
        self.alpha_bar_t1 = torch.tensor(alpha_bar_t1, device=device).float()
        self.alpha_bar_t2 = torch.tensor(alpha_bar_t2, device=device).float()
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # x: batch of images normalized [-1,1]
        # sample noisy versions xt1 and xt2
        # q(xt|x0) = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        def get_xt(x, t):
            alpha_bar = t.view(-1, 1, 1, 1)
            eps = torch.randn_like(x)
            return (alpha_bar ** 0.5) * x + ((1 - alpha_bar) ** 0.5) * eps

        batch_size = x.shape[0]
        t1_tensor = self.alpha_bar_t1.expand(batch_size)
        t2_tensor = self.alpha_bar_t2.expand(batch_size)
        xt1 = get_xt(x, t1_tensor)
        xt2 = get_xt(x, t2_tensor)
        h1 = self.model(xt1, t1_tensor)
        h2 = self.model(xt2, t2_tensor)
        return h1, h2


# -----------------------------------------------------------------------------
# Content Integration Module (CIM)
# -----------------------------------------------------------------------------
class CIM(nn.Module):
    def __init__(self, in_channels, h_channels, heads=4):
        super().__init__()
        assert in_channels % heads == 0, "in_channels must be divisible by heads"
        self.dim_head = in_channels // heads
        self.ht1_conv = nn.Conv2d(h_channels, in_channels, 1)
        self.norm_q = nn.BatchNorm2d(in_channels)
        self.norm_k = nn.BatchNorm2d(in_channels)
        self.norm_v = nn.BatchNorm2d(in_channels)
        self.to_q = nn.Conv2d(in_channels, in_channels, 1)
        self.to_k = nn.Conv2d(in_channels, in_channels, 1)
        self.to_v = nn.Conv2d(in_channels, in_channels, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)
        self.heads = heads
        self.scale = self.dim_head ** -0.5

    def forward(self, f, h):
        """
        f: Tensor of shape (B, C, H, W)
        h: Tensor of shape (B, C_h, H_h, W_h)
        returns: Tensor of shape (B, C, H, W)
        """
        b, c, h_, w_ = f.shape
        h_aligned = self.ht1_conv(h)
        q = self.to_q(self.norm_q(f)).reshape(b, self.heads, self.dim_head, h_ * w_)
        k = self.to_k(self.norm_k(h_aligned)).reshape(b, self.heads, self.dim_head,-1)
        v = self.to_v(self.norm_v(h_aligned)).reshape(b, self.heads, self.dim_head,-1)
        attn = torch.softmax(torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale, dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, c, h_, w_)
        return f + self.to_out(out)


# -----------------------------------------------------------------------------
# Haze-Aware Enhancement Module (HAE)
# -----------------------------------------------------------------------------
class HAE(nn.Module):
    def __init__(self, channels, h_channels, reduction=16):
        super().__init__()
        self.init_conv = nn.Conv2d(h_channels, h_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.gamma_conv1 = nn.Conv2d(h_channels, channels, kernel_size=1, stride=2)
        self.gamma_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.beta_conv1 = nn.Conv2d(h_channels, channels, kernel_size=1, stride=2)
        self.beta_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, f, h):
        while f.shape[2] != h.shape[2]:
            h = nn.MaxPool2d(2)(h)
        h = self.init_conv(h)
        h = self.upsample(h)
        gamma = self.gamma_conv1(h)
        gamma = self.gamma_conv2(gamma)
        beta = self.beta_conv1(h)
        beta = self.beta_conv2(beta)
        out = gamma * f + beta
        w = self.pool(out).view(out.size(0), -1)
        w = torch.sigmoid(self.fc(w)).view(out.size(0), out.size(1), 1, 1)
        return out * w + f


# -----------------------------------------------------------------------------
# DiffLI2D Block and U-Net
# -----------------------------------------------------------------------------
class DiffLI2DBlock(nn.Module):
    def __init__(self, channels, h_channels):
        super().__init__()
        self.cim = CIM(channels, h_channels)
        self.hae = HAE(channels, h_channels)

    def forward(self, x, h1, h2):
        out = self.cim(x, h1)
        out = self.hae(out, h2)
        return out


class DiffLI2D(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, levels=4, h_channels=6):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # down
        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(levels - 1):
            self.downs.append(DiffLI2DBlock(ch, h_channels))
            self.downs.append(nn.AvgPool2d(kernel_size=2, stride=2))
        # bottleneck layer
        self.bot = DiffLI2DBlock(ch, h_channels)
        # up
        self.ups = nn.ModuleList()
        for _ in range(levels - 1):
            self.ups.append(nn.ConvTranspose2d(ch, ch, 2, stride=2))
            self.ups.append(DiffLI2DBlock(ch, h_channels))
        self.concat_convL3 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.concat_convL2 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.concat_convL1 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, h1, h2):
        skips = []
        out = self.init_conv(x)
        # down
        for layer in self.downs:
            if isinstance(layer, DiffLI2DBlock):
                out = layer(out, h1, h2)
                skips.append(out)
            else:
                out = layer(out)
        # bottleneck layer
        out = self.bot(out, h1, h2)
        # up
        flag = 3
        for layer in self.ups:
            if isinstance(layer, nn.ConvTranspose2d):
                out = layer(out)
            else:
                skip = skips.pop()
                out = torch.cat([out, skip], dim=1)
                if flag == 3:
                    out = self.concat_convL3(out)
                    flag = flag - 1
                elif flag == 2:
                    out = self.concat_convL2(out)
                    flag = flag - 1
                elif flag == 1:
                    out = self.concat_convL1(out)
                    flag = flag - 1
                out = layer(out, h1, h2)
        out = self.final_conv(out)
        return out + x
