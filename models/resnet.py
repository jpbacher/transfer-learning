import torch.nn as nn
from torchvision.models import resnet50


def resnet_finetune(model):
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(2048, 1024),
                             nn.ReLU(),
                             nn.Linear(1024, 512),
                             nn.ReLU(),
                             nn.Linear(512, 1),
                             nn.Sigmoid())
    return model


resnet50 = resnet_finetune(resnet50(pretrained=True))
