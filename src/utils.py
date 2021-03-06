import numpy as np
import matplotlib.pyplot as plt
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_img(img):
    img = plt.imread(img)
    plt.imshow(img)
    print(f'Shape: {np.array(img).shape}')


def plot_tensor(img, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # set color channel to 3rd dim
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    plt.axis('off')
    return ax, img


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
