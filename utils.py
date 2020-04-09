import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_img(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
    plt.show()

def show_batch(dataset, n=10):
    imgs = [dataset[img][0] for img in range(n)]
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()
