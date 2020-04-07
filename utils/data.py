import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


class CucumberZucchini(Dataset):
    def __init__(self, data_path, val_size, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.train_loader, self.val_loader = self._get_loaders(self.data_path, self.val_size, self.transform)

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

    def __len__(self):
        return len(self.train_loader)
