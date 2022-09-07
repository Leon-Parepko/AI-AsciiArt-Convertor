import torch
from torch.utils.data import Dataset
from skimage import io
import os

class SymbolDataset(Dataset):
    def __init__(self, root_img_dir, transform=None):
        self.root_img_dir = root_img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_img_dir))

    def __getitem__(self, index, class_view="numeric"):
        img_path = os.path.join(self.root_img_dir, f"{index}.jpg")
        x = io.imread(img_path)

        if class_view == "numeric":
            y = torch.tensor(int(index))

        if self.transform:
            print(type(x))
            x = self.transform(x)
            print(type(x))
        return (x, y)