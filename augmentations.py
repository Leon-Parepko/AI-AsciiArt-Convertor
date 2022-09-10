import numpy as np
import torchvision.transforms as tr

final_transform = tr.Compose([
    tr.ToPILImage(),
    # tr.ColorJitter(brightness=1),
    tr.RandomRotation(degrees=15),
    tr.RandomPerspective(),
    # tr.RandomAffine(degrees=30),
    # tr.RandomAdjustSharpness(sharpness_factor=3),
    # tr.GaussianBlur(kernel_size=9),
])


symbol_transform = tr.Compose([
    tr.ToPILImage(),
    tr.Resize(42),
    tr.RandomCrop((47, 27)),
    tr.RandomAdjustSharpness(sharpness_factor=np.random.uniform(low=0, high=8), p=0.8),
    # tr.GaussianBlur(kernel_size=17),
    tr.ToTensor()
    # tr.ColorJitter(brightness=1),

    # tr.RandomRotation(degrees=30),
    # tr.RandomHorizontalFlip(),
    # tr.RandomVerticalFlip(),
    # tr.RandomPerspective(),
    # # tr.RandomAffine(degrees=30),
    # tr.RandomAdjustSharpness(sharpness_factor=3),
])


background_transform = tr.Compose([
    tr.ToPILImage(),
    tr.GaussianBlur(kernel_size=27),
    # tr.ColorJitter(brightness=1),
    tr.ToTensor()
])