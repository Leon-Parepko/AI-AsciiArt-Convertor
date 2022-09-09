import torch
from torch.utils.data import Dataset
from skimage import io
import os
import random

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
            x = self.transform(x)

        return x, y



class NaturalImagesDataset(Dataset):
    def __init__(self, root_img_dir, transform=None):
        self.root_img_dir = root_img_dir
        self.transform = transform
        self.obj_classes = os.listdir(self.root_img_dir)

    def __len__(self, obj_type):
        return len(os.listdir(os.path.join(self.root_img_dir, obj_type)))

    def __getitem__(self, index, obj_class="random"):
        if obj_class == "random":
            y = self.obj_classes[random.randint(0, len(self.obj_classes) - 1)]
            img_path = os.path.join(self.root_img_dir, y, f"{y}_{index:04d}.jpg")
            x = io.imread(img_path)

        else:
            y = obj_class
            img_path = os.path.join(self.root_img_dir, y, f"{y}_{index:04d}.jpg")
            x = io.imread(img_path)

        if self.transform:
            x = self.transform(x)

        return x, y