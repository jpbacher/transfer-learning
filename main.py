import torch.optim as optim
import torch.nn as nn

from project import Project
from data import get_loaders
from data.transformation import data_transforms
from models.pretrained import get_pretrained_model
from train import train
from logger import logging
from src.utils import device


if __name__ == '__main__':
    project = Project()
    logging.info(f'*** Using device: {device}')
    params = {
        'lr': 0.001,
        'train_test_split': 0.85,
        'train_val_split': 0.15,
        'batch_size': 16,
        'epochs': 20,
        'patience': 5,
        'model': 'resnet50',
        'n_classes': 4
    }
    train_dl, val_dl, test_dl = get_loaders(
        project.data_dir,
        train_transforms=data_transforms['train'],
        val_transforms=data_transforms['val'],
        train_test_split=params['train_test_split'],
        train_val_split=params['train_val_split'],
        batch_size=params['batch_size']
    )
    model = get_pretrained_model(params['model'], params['n_classes'])
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.NLLLoss()
    model, history = train(
        model, loss_fn, optimizer, train_dl, val_dl, 'saved_models/resnet-weather.pt',
        n_epochs=params['epochs'], patience=params['patience']
    )
