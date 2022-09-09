# import torch.nn as nn
import torchvision.transforms as tr

transforms = tr.Compose([
    tr.ToPILImage(),
    # tr.RandomCrop((47, 27)),
    tr.ColorJitter(brightness=1),
    tr.RandomRotation(degrees=30),
    tr.RandomHorizontalFlip(),
    tr.RandomVerticalFlip(),
    tr.RandomPerspective(),
    # tr.RandomAffine(degrees=30),
    tr.RandomAdjustSharpness(sharpness_factor=3),
    tr.GaussianBlur(kernel_size=9),
])


symbol_transform = tr.Compose([
    tr.ToPILImage(),
    tr.Resize(37),
    tr.RandomCrop((47, 27)),
    tr.GaussianBlur(kernel_size=21),
    tr.ToTensor()
    # tr.ColorJitter(brightness=1),

    # tr.RandomRotation(degrees=30),
    # tr.RandomHorizontalFlip(),
    # tr.RandomVerticalFlip(),
    # tr.RandomPerspective(),
    # # tr.RandomAffine(degrees=30),
    # tr.RandomAdjustSharpness(sharpness_factor=3),


])