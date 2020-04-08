import glob
from PIL import Image
from torch.utils.data import Dataset


class CucumberZucchini(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.img_files = [img for img in glob.glob(self.data_dir + '**/*/**/*', recursive=True)]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __get__item(self, idx):
        print(f'Retrieving image {idx}')
        image = Image.open(self.img_files[idx])
        return image
