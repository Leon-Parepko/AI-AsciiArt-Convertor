import os
import numpy as np
import torch
import torchvision

import augmentations
from skimage import io


def gpu_check():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_files(dir):
    counter = 0
    for content in os.walk(dir):
        files = content[2]
        counter += len(files)
    return counter


def random_img_split(img):
    h, w = img.shape[0], img.shape[1]
    random_point = np.random.uniform(low=0, high=1)
    h_split, w_split = round(h * random_point), round(w * random_point)

    if h_split - 47 <= 0:
        h_split = h_split + abs(h_split - 48)

    if w_split - 27 <= 0:
        w_split = w_split + abs(w_split - 28)

    new_img = img[h_split - 47:h_split, w_split - 27:w_split]
    return new_img


def combine_images(bg, symb):
    bg_crop = random_img_split(bg)
    augmented_symb = augmentations.symbol_transform(symb).reshape((47, 27))
    augmented_bg_small = augmentations.background_transform(bg_crop).reshape((47, 27))

    bg_small_avg_color = torch.mean(bg_crop)

    if torch.round(bg_small_avg_color) == 0:
        tone_mult = 1

    else:
        tone_mult = 0

    # Actually this one somehow combines these two images
    main_img = (augmented_symb * augmented_bg_small - 0.5) + augmented_symb * (-tone_mult)
    augmented_main_image = augmentations.final_transform(main_img)

    return augmented_main_image


def load_image(path):
    img = torchvision.transforms.ToTensor()(io.imread(path))
    return img