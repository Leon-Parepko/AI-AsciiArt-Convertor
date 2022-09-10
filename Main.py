
"""
    This program could convert any image to
        it's corresponding Ascii-Art using
        convolutional neural network (as symbol
        classifier).
"""

# Import all libraries
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import checks
import Functional
import augmentations
from Models import cnn_model
from datasets import SymbolDataset
from datasets import NaturalImagesDataset
import matplotlib.pyplot as plt

# # Load image
# img = ...
#
# # Configure global variables
# used_symbols = [...]
mode = "train"
# width, hight, depth = img.shape
# img_shape = [width, hight, depth]





model = cnn_model().to(checks.gpu_check())

if mode == "train":
    # Load all images as datasets (without augmentation)
    symb_dataset = SymbolDataset("Data/Dataset/Font_letters", transform=torchvision.transforms.ToTensor())
    img_dataset = NaturalImagesDataset("Data/Dataset/Natural_Images", transform=torchvision.transforms.ToTensor())

    symb = symb_dataset[16][0].reshape((47, 27))
    bg = torchvision.transforms.Grayscale(num_output_channels=1)(img_dataset[10][0])[0]
    bg_small = Functional.random_img_split(bg)

    augmented_symb = augmentations.symbol_transform(symb).reshape((47, 27))
    augmented_bg_small = augmentations.background_transform(bg_small).reshape((47, 27))

    bg_small_avg_color = torch.mean(bg_small)

    if torch.round(bg_small_avg_color) == 0:
        tone_mult = 1

    else:
        tone_mult = 0

    # Actually this one somehow combines these two images
    main_img = (augmented_symb * augmented_bg_small - 0.5) + augmented_symb * (-tone_mult)

    # plt.imshow(transforms(symb_dataset[10][0].reshape((47, 27))), cmap='Greys')
    # plt.imshow(transforms(img_dataset.__getitem__(160, obj_class="cat")[0]), cmap='Greys')
    plt.imshow(augmentations.final_transform(main_img), cmap='Greys')
    plt.show()


    epochs = 2
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(0, epochs):
        pass















    #
    # with torch.no_grad:
    #     pass
    # for epoch in range(0, epochs):
    #     for i, (features, ans) in enumerate(tqdm(train_batch_loader)):
    #         # Forward pass
    #         Y_pred = mnist_nn.forward(features.view(-1, 28 * 28))
    #
    #         loss = loss_func(Y_pred, ans)
    #
    #         # Backward pass
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    #         if i % 3 == 0: history.append(loss.data)