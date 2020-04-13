import numpy as np
import torchvision.transforms as T


data_transforms = {
    'train': T.Compose([
        T.RandomResizedCrop(size=256, scale=(0.8,1.0)),
        T.RandomRotation(degrees=15),
        T.ColorJitter(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
