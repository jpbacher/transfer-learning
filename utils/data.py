import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


class CucumberZucchini(Dataset):
    def __init__(self, data_root, transforms=None):
        self.data_root = data_root
        self.img_files = [img for img in glob.glob(self.data_root + '**/*/**/*', recursive=True)]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __get__item(self, idx):
        print(f'Retrieving image {idx}')
        image = Image.open(self.img_files[idx])
        return image

     def _get_loaders(self, data_dir, val_size, transforms):
        train_sample, val_sample = self._get_samplers(val_size, transforms)
        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=8, sampler=train_sample)
        val_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=8, sampler=val_sample)
        return train_loader, val_loader

    def _get_samplers(self, val_size, transforms):
        np.random.seed(30)
        train_count = len(self.train_data)
        indices = list(range(train_count))
        split = int(train_count * val_size)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sample = SubsetRandomSampler(train_idx)
        val_sample = SubsetRandomSampler(val_idx)
        return train_sample, val_sample

    self.train_data = datasets.ImageFolder(self.data_root, transforms)
    self.val_data = datasets.ImageFolder(self.data_root, transforms)
    self.train_loader, self.val_loader = self._get_loaders(self.img_files, self.val_size, transforms)

    def train_model(self, model, loaders, loss_fn, optimizer, epochs):
