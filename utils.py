import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_img(img):
    img = plt.imread(img)
    plt.imshow(img)


def plot_loss_results(history):
    for loss in ['train_loss', 'val_loss']:
        plt.figure(figsize=(10, 8))
        plt.plot(history[loss], label=loss)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()