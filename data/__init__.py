import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler

from data.transformation import data_transforms


def get_loaders(
        data_dir,
        train_transforms=None,
        val_transforms=None,
        train_test_split=0.85,
        train_val_split=0.15,
        batch_size=16,
        shuffle=True):
    """
    This function returns the training, validation, & test loaders.
    """
    np.random.seed(24)
    train_ds = ImageFolder(root=data_dir, transform=train_transforms)
    val_ds = ImageFolder(root=data_dir, transform=val_transforms)
    test_ds = ImageFolder(root=data_dir, transform=val_transforms)
    img_count = len(train_ds)
    indices = list(range(img_count))
    test_split = int(img_count * train_test_split)
    if shuffle:
        np.random.shuffle(indices)
    train_idx, test_idx = indices[:test_split], indices[test_split:]
    train_count = len(train_idx)
    val_split = int(train_count * (1 - train_val_split))
    train_idx, val_idx = train_idx[:val_split], train_idx[val_split:]
    train_sample = SubsetRandomSampler(train_idx)
    val_sample = SubsetRandomSampler(val_idx)
    test_sample = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sample)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sample)
    test_loader = DataLoader(test_ds, batch_size=batch_size, sampler=test_sample)

    return train_loader, val_loader, test_loader
