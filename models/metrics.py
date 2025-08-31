# metrics.py
import torch
import torchvision.transforms.functional as TF
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips 


lpips_alex = lpips.LPIPS(net='alex')  # 选择 AlexNet 版 LPIPS
lpips_alex.eval()


def PSNR(pred, target):
    pred = pred.float()
    target = target.float()
    pred_01 = (pred + 1) / 2
    tgt_01  = (target + 1) / 2
    return psnr(pred_01, tgt_01, data_range=1.0).item()


def Y_PSNR(pred, target):
    pred = pred.float()
    target = target.float()
    pred_01 = (pred + 1.0) / 2.0
    tgt_01  = (target + 1.0) / 2.0
    r_p, g_p, b_p = pred_01[:, 0:1, :, :], pred_01[:, 1:2, :, :], pred_01[:, 2:3, :, :]
    r_t, g_t, b_t = tgt_01[:, 0:1, :, :], tgt_01[:, 1:2, :, :], tgt_01[:, 2:3, :, :]
    y_p = 0.299 * r_p + 0.587 * g_p + 0.114 * b_p
    y_t = 0.299 * r_t + 0.587 * g_t + 0.114 * b_t
    mse = torch.mean((y_p - y_t) ** 2)
    if mse == 0:
        return float('inf')
    psnr_val = 10.0 * torch.log10(1.0 / mse)
    return psnr_val.item()


def SSIM(pred, target):
    pred = pred.float()
    pred = pred.float()
    target = target.float()
    pred_01 = (pred + 1) / 2
    tgt_01  = (target + 1) / 2
    return ssim(pred_01, tgt_01, data_range=1.0).item()


def LPIPS(pred, target):
    pred = pred.float()
    device = pred.device
    lpips_model = lpips_alex.to(device)
    with torch.no_grad():
        lpips_val = lpips_model(pred, target).mean().item()
    return lpips_val
