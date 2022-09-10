import math
import torch
import torchvision
from torch.utils.data import Dataset
from skimage import io
import os
import Functional


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
    def __init__(self, root_img_dir, obj_class="", transform=None):
        self.root_img_dir = root_img_dir
        self.obj_class = obj_class
        self.transform = transform
        self.obj_classes = os.listdir(self.root_img_dir)
        self.every_class_length = []
        for cl in self.obj_classes:
            self.every_class_length.append(Functional.count_files(os.path.join(self.root_img_dir, cl)))


    def __len__(self):
        if self.obj_class == "":
            length = sum(self.every_class_length)
        else:
            length = self.every_class_length[self.obj_classes.index(self.obj_class)]
        return length


    def __getitem__(self, index):
        if self.obj_class == "":
            y = ''
            for (class_len, cl) in zip(self.every_class_length, self.obj_classes):
                index -= class_len
                if index < 0:
                    y = cl
                    index = abs(index) - 1
                    break
                elif index == 0:
                    y = cl
                    break
                index -= 1

            img_path = os.path.join(self.root_img_dir, y, f"{y}_{index:04d}.jpg")
            x = io.imread(img_path)

        else:
            y = self.obj_class
            img_path = os.path.join(self.root_img_dir, y, f"{y}_{index:04d}.jpg")
            x = io.imread(img_path)

        if self.transform:
            x = self.transform(x)

        return x, y


class AugmentedDataset(Dataset):
    def __init__(self, bg_obj_class="", transform=None):
        # Load all images as datasets (without augmentation)
        self.symb_dataset = SymbolDataset("Data/Dataset/Font_letters", transform=torchvision.transforms.ToTensor())
        self.bg_dataset = NaturalImagesDataset("Data/Dataset/Natural_Images", obj_class=bg_obj_class,  transform=torchvision.transforms.ToTensor())
        self.transform = transform


    def __len__(self):
        return (len(self.bg_dataset) + 0) * (len(self.symb_dataset) + 0)


    def __getitem__(self, index):
        sym_index = math.floor(index / len(self.bg_dataset))
        bg_index = index % len(self.bg_dataset)

        symb = self.symb_dataset[sym_index][0].reshape((47, 27))
        bg = torchvision.transforms.Grayscale(num_output_channels=1)(self.bg_dataset[bg_index][0])[0]

        x = Functional.combine_images(bg, symb)
        y = self.symb_dataset[sym_index][1]

        return x, y
