import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_img(img):
    img = plt.imread(img)
    plt.imshow(img)


def plot_loss_results(history):
    plt.figure(figsize=(10, 8))
    for loss in ['train_loss', 'val_loss']:
        plt.plot(history[loss], label=loss)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.show()


def plot_accuracy_results(history):
    plt.figure(figsize=(10, 8))
    for acc in ['train_acc', 'val_acc']:
        plt.plot(history[acc], label=acc)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.show()
