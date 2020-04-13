import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.utils import make_grid

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


def process_to_tensor(img_path):
    """
    Processes an image path into PyTorch tensor. Applies same transformations that was
    done to the validation set.
    :param img_path: path to an image
    :return: PyTorch tensor
    """
    img = Image.open(img_path)
    # resize image
    img = img.resize((256, 256))
    # center crop
    w, h = 256, 256
    new_w, new_h = 224, 224
    left = (w - new_w) / 2
    right = (w + new_w) / 2
    top = (h - new_h) / 2
    bottom = (h + new_h) / 2
    img = img.crop((left, top, right, bottom))
    # transpose color dimension & normalize
    img = np.array(img).transpose((2, 0, 1))
    img = img / 256
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = img - mean
    img = img / std
    img_tensor = torch.Tensor(img)
    return img_tensor


def predict(img_path, model, top_k=2):
    """
    Make prediction on an unseen image using pre-trained model.
    :param img_path: filename on image
    :param model: PyTorch model for inference
    :param top_k: the number of top predictions to return
    :return: top probabilities & top classes
    """
    img_tensor = process_to_tensor(img_path)
    img_tensor = img_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        output = model.forward(img_tensor)
        pred = torch.exp(output)
        topk, top_labels = pred.topk(top_k, dim=1)
        # top_labels (convert to actual classes)
        top_prob = topk.numpy()[0]
    return top_prob, top_labels
