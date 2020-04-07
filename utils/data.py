import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


class CucumberZucchini(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.img_files = [img for img in glob.glob(self.data_root + '**/*/**/*', recursive=True)]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __get__item(self, idx):
        print(f'Retrieving image {idx}')
        image = Image.open(self.img_files[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image

     def _get_loaders(self, data_dir, val_size, transforms):
        np.random.seed(30)
        train_data = datasets.ImageFolder(data_dir, transforms)
        val_data = datasets.ImageFolder(data_dir, transforms)
        train_count = len(train_data)
        indices = list(range(train_count))
        split = int(train_count * val_size)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sample = SubsetRandomSampler(train_idx)
        val_sample = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=8, sampler=train_sample)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=8, sampler=val_sample)
        return train_loader, val_loader

