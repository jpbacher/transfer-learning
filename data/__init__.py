import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler

from data.transformation import data_transforms


def get_loaders(
        data_dir,
        train_transforms=None,
        val_transforms=None,
        val_size=0.2,
        batch_size=16):
    """
    This function returns the training & validation loaders.

    :param data_dir:
    :param train_transforms:
    :param val_transforms:
    :param val_size:
    :param batch_size:
    :return loaders
    """
    np.random.seed(24)
    train_data = ImageFolder(root=data_dir, transform=train_transforms)
    val_data = ImageFolder(root=data_dir, transform=val_transforms)
    length = len(train_data)
    indices = list(range(length))
    split = int(length * val_size)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sample, val_sample = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sample)
    val_loader = DataLoader(val_data, batch_size=batch_size, sampler=val_sample)

    return train_loader, val_loader
