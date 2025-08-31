# restore.py
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from guided_diffusion import dist_util
from models.metrics import PSNR, SSIM, LPIPS, Y_PSNR
import torch.distributed as dist
from lpips import LPIPS as LPIPSModule


class Trainer:
    def __init__(self, model, extractor, optimizer, scheduler,
                 train_loader, val_loader, test_loader,
                 device, work_dir, total_epochs, save_every=100):
        self.model = model.to(device)
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.work_dir = work_dir
        self.criterion = nn.L1Loss()
        self.best_psnr = 0.0
        self.save_every = save_every
        self.total_epochs = total_epochs
        self.start_epoch = 1
        self.lpips_alex = LPIPSModule(net='alex', version='0.1').to(self.device)
        os.makedirs(work_dir, exist_ok=True)

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt.get('epoch', 1) + 1
        self.best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"Loaded checkpoint from {ckpt_path}")

    def train(self, train_sampler):
        self.train_sampler = train_sampler
        self.model.train()
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.train_sampler.set_epoch(epoch)
            total_loss = 0.0
            for hazy, clean in self.train_loader:
                hazy, clean = hazy.to(self.device), clean.to(self.device)
                h1, h2 = self.extractor(hazy)  # extract h-space feature
                output = self.model(hazy, h1, h2)
                loss = self.criterion(output, clean)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()

            if dist.get_rank() == 0:
                avg_loss = total_loss / len(self.train_loader)
                lr = self.optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch}] LR={lr:.2e}  TrainLoss={avg_loss:.4f}")

                if epoch % self.save_every == 0 or epoch == self.total_epochs:
                    metrics = self.validate(epoch)
                    self._maybe_save_checkpoint(epoch, metrics['psnr'])

    def validate(self, epoch):
        self.model.eval()
        losses, psnrs, ssims, lpips, ypsnrs = [], [], [], [], []
        with torch.no_grad():
            for i, (hazy, clean) in enumerate(self.val_loader):
                hazy, clean = hazy.to(self.device), clean.to(self.device)
                h1, h2 = self.extractor(hazy)
                output = self.model(hazy, h1, h2).clamp(-1.0, 1.0)
                losses.append(self.criterion(output, clean).item())
                psnrs.append(PSNR(output, clean))
                ssims.append(SSIM(output, clean))
                lpips.append(LPIPS(output, clean))
                ypsnrs.append(Y_PSNR(output, clean))

                if i == 0 and dist.get_rank() == 0:
                    save_dir = os.path.join(self.work_dir, 'val-image')
                    os.makedirs(save_dir, exist_ok=True)
                    hazy_row = hazy[:4]
                    out_row = output[:4]
                    clean_row = clean[:4]
                    grid_imgs = torch.cat([hazy_row, out_row, clean_row], dim=0)
                    save_image(grid_imgs, os.path.join(save_dir, f"val_out_{epoch}.png"),
                               normalize=True, nrow=hazy_row.size(0))

        metrics = {
            'loss': sum(losses) / len(losses),
            'psnr': sum(psnrs) / len(psnrs),
            'ssim': sum(ssims) / len(ssims),
            'lpips': sum(lpips) / len(lpips),
            'y-psnr': sum(ypsnrs) / len(ypsnrs),
        }
        if dist.get_rank() == 0:
            print(f"[Val {epoch}] Loss={metrics['loss']:.4f}  PSNR={metrics['psnr']:.2f}  "
                  f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f} Y-PSNR={metrics['y-psnr']:.2f}")
        return metrics

    def test(self):
        self.model.eval()
        losses, psnrs, ssims, lpips = [], [], [], []
        save_dir = os.path.join(self.work_dir, 'test-image')
        os.makedirs(save_dir, exist_ok=True)
        with torch.no_grad():
            for idx, (hazy, clean) in enumerate(self.test_loader):
                hazy, clean = hazy.to(self.device), clean.to(self.device)
                h1, h2 = self.extractor(hazy)
                output = self.model(hazy, h1, h2).clamp(-1.0, 1.0)
                save_image(output, os.path.join(save_dir, f"test_out_{idx}.png"), normalize=True)
                losses.append(self.criterion(output, clean).item())
                psnrs.append(PSNR(output, clean))
                ssims.append(SSIM(output, clean))
                lpips.append(LPIPS(output, clean))
        metrics = {
            'loss': sum(losses) / len(losses),
            'psnr': sum(psnrs) / len(psnrs),
            'ssim': sum(ssims) / len(ssims),
            'lpips': sum(lpips) / len(lpips),
        }
        if dist.get_rank() == 0:
            print(f"[Test] Loss={metrics['loss']:.4f}  PSNR={metrics['psnr']:.2f}  "
                  f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}")
        return metrics

    def _maybe_save_checkpoint(self, epoch, val_psnr):
        if dist.get_rank() != 0:
            return
        pt_save_dir = os.path.join(self.work_dir, "checkpoint")
        os.makedirs(pt_save_dir, exist_ok=True)
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_psnr': val_psnr
        }
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            best_path = os.path.join(pt_save_dir, f"best-epoch={epoch}-val_PSNR={val_psnr:.2f}.pth")
            torch.save(ckpt, best_path)
            print(f"→ New best PSNR {val_psnr:.2f}, checkpoint saved to {best_path}.")
        if epoch % 10 == 0:
            ten_path = os.path.join(pt_save_dir, f"epoch={epoch}-val_PSNR={val_psnr:.2f}.pth")
            torch.save(ckpt, ten_path)
            print(f"→ every ten PSNR {val_psnr:.2f}, checkpoint saved to {ten_path}.")
        if epoch == self.total_epochs:
            last_path = os.path.join(pt_save_dir, "last.pth")
            torch.save(ckpt, last_path)
            print(f"→ last checkpoint saved to {last_path}.")
