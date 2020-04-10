import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_img(img):
    img = plt.imread(img)
    plt.imshow(img)
