import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets.folder import ImageFolder
from PIL import Image


def get_labels(model, data_dir):
    train_ds = ImageFolder(root=data_dir)
    model.idx_to_class = train_ds.class_to_idx
    model.idx_to_class = {
        idx: label for label, idx in model.idx_to_class.items()
    }
    return model.idx_to_class


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


def predict(data_dir, img_path, model, top_k=2):
    """
    Make prediction on an unseen image using pre-trained model.
    :param data_dir: directory where images located
    :param img_path: filename on image
    :param model: PyTorch model for inference
    :param top_k: the number of top predictions to return
    :return: top probabilities & top classes
    """
    img_tensor = process_to_tensor(img_path)
    img_tensor = img_tensor.view(1, 3, 224, 224)
    model.class_to_idx = get_labels(model, data_dir)
    with torch.no_grad():
        model.eval()
        output = model.forward(img_tensor)
        pred = torch.exp(output)
        topk, top_labels = pred.topk(top_k, dim=1)
        top_labels = [
            model.idx_to_class[label] for label in top_labels.numpy()[0]
        ]
        top_prob = topk.numpy()[0]
    return top_prob, top_labels
