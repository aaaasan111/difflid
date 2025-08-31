import torch 
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, utils
from torchvision.transforms import functional as TF, RandomCrop
from PIL import Image
import os
import random
from torch.utils.data.distributed import DistributedSampler


class HazyDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir, resize=128,transform=None):
        self.hazy_paths = sorted([os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir)])
        self.clean_paths = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)])
        self.transform = transform
        self.resize = resize
    
    def __len__(self):
        return len(self.hazy_paths)
    
    def __getitem__(self, idx):
        hazy = Image.open(self.hazy_paths[idx]).convert('RGB')
        clean = Image.open(self.clean_paths[idx]).convert('RGB')

        if random.random() < 0.5:
            hazy = TF.hflip(hazy)
            clean = TF.hflip(clean)

        i, j, h, w = RandomCrop.get_params(hazy, output_size=(self.resize, self.resize))
        hazy = TF.crop(hazy, i, j, h, w)
        clean = TF.crop(clean, i, j, h, w)

        if self.transform:
            hazy = self.transform(hazy) 
            clean = self.transform(clean) 
        return hazy, clean


# 读入txt文件中的数据路径
class PairsDataset(Dataset):
    def __init__(self, pairs_txt, resize=128, transform=None):
        self.pairs_txt=pairs_txt
        with open(pairs_txt, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.pairs = [tuple(line.split('|')) for line in lines]
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        haze_path, clean_path = self.pairs[idx]
        hazy = Image.open(haze_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        if random.random() < 0.5:
            hazy = TF.hflip(hazy)
            clean = TF.hflip(clean)
        i, j, h, w = RandomCrop.get_params(hazy, output_size=(self.resize, self.resize))
        hazy = TF.crop(hazy, i, j, h, w)
        clean = TF.crop(clean, i, j, h, w)
        if self.transform:
            hazy = self.transform(hazy)
            clean = self.transform(clean)
        return hazy, clean


def make_dataloader(hazy_dir, clean_dir, batch_size, num_workers=4,resize=128,shuffle=True,pin_memory=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    # 根据路径类型选择数据集模式
    if isinstance(hazy_dir, str) and hazy_dir.lower().endswith('.txt') and os.path.isfile(hazy_dir):
        dataset = PairsDataset(pairs_txt=hazy_dir, resize=resize, transform=transform)
    else:
        dataset = HazyDataset(hazy_dir, clean_dir, resize, transform)
    data_sampler=DistributedSampler(dataset,shuffle=shuffle)
    loader  = DataLoader(dataset, batch_size=batch_size,sampler=data_sampler,
                         num_workers=num_workers, pin_memory=pin_memory)
    return loader,data_sampler