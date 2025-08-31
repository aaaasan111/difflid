# train.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    create_model_and_diffusion, model_and_diffusion_defaults,
    add_dict_to_argparser, args_to_dict
)
from data.dataset import make_dataloader
from models.net import HSpaceExtractor, DiffLI2D
from restore import Trainer
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def create_ddpm_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",  # Pretrained DDPM model weight file path
        classifier_path="",  # Pretrained classifier model weight file path
        classifier_scale=1.0,
        use_fp16=False,

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def create_difflid_args():
    parser = argparse.ArgumentParser('Train the diffli2d_restore module')
    parser.add_argument('--hazy_dir', type=str, default='', help='Haze Image Folder Path')
    parser.add_argument('--clean_dir', type=str, default='', help='Clean image folder path')
    parser.add_argument('--val_hazy_dir', type=str, default='', help='Validation Set haze Image Folder Path')
    parser.add_argument('--val_clean_dir', type=str, default='', help='Validation Set clean Image Folder Path')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader worker threads')
    # Training parameters
    parser.add_argument('--resize', type=int, default=64, help='Input image resize size')
    parser.add_argument('--batch_size', type=int, default=7)
    parser.add_argument('--total_epochs', type=int, default=1200, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--save_every', type=int, default=1, help='Save validation results and logs every N epochs')
    parser.add_argument('--work_dir', type=str, default='',help='Directory to save results and model weights')
    parser.add_argument('--resume_from', type=str,default="",help='checkpoint path')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--t1', type=int, default=0, help='First diffusion timestep t1')
    parser.add_argument('--t2', type=int, default=500, help='Second diffusion timestep t2')
    args = parser

    return args


def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    ddpm_parser = create_ddpm_argparser()
    ddpm_args, rest = ddpm_parser.parse_known_args()
    difflid_parser = create_difflid_args()
    difflid_args = difflid_parser.parse_args(rest)

    # set seed
    random.seed(difflid_args.seed)
    np.random.seed(difflid_args.seed)
    torch.manual_seed(difflid_args.seed)
    torch.cuda.manual_seed_all(difflid_args.seed)

    model_d, diffusion = create_model_and_diffusion(**args_to_dict(ddpm_args, model_and_diffusion_defaults().keys()))
    model_d.load_state_dict(dist_util.load_state_dict(ddpm_args.model_path, map_location=dist_util.dev()))
    model_d.eval()
    alphas = diffusion.alphas_cumprod
    alpha_bar_t1 = alphas[difflid_args.t1]  # Get alpha_bar for timestep t1 and t2
    alpha_bar_t2 = alphas[difflid_args.t2]
    extractor = HSpaceExtractor(model_d, alpha_bar_t1, alpha_bar_t2, device)
    extractor.model = DDP(extractor.model, device_ids=[local_rank])

    # Build dehazing network, optimizer, and scheduler
    net = DiffLI2D(in_channels=3, base_channels=64, levels=4, h_channels=6).to(device)
    net = DDP(net, device_ids=[local_rank])
    optimizer = torch.optim.Adam(net.parameters(), lr=difflid_args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    criterion = nn.L1Loss()

    # Build DataLoaders
    train_loader, train_sampler = make_dataloader(difflid_args.hazy_dir, difflid_args.clean_dir,
                                                  difflid_args.batch_size, difflid_args.num_workers,
                                                  difflid_args.resize, difflid_args.shuffle, difflid_args.pin_memory)
    val_loader, _ = make_dataloader(difflid_args.val_hazy_dir, difflid_args.val_clean_dir, difflid_args.batch_size,
                                    difflid_args.num_workers, difflid_args.resize, False, difflid_args.pin_memory)
    test_loader, _ = make_dataloader(difflid_args.val_hazy_dir, difflid_args.val_clean_dir, difflid_args.batch_size, difflid_args.num_workers,
                                     difflid_args.resize, False, difflid_args.pin_memory)

    trainer = Trainer(net, extractor, optimizer, scheduler,
                      train_loader, val_loader, test_loader,
                      device, difflid_args.work_dir, total_epochs=difflid_args.total_epochs,
                      save_every=difflid_args.save_every)

    if difflid_args.resume_from:
        trainer.load_checkpoint(difflid_args.resume_from)
    trainer.train(train_sampler)
    # trainer.test()

    # Cleanup distributed process group
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
