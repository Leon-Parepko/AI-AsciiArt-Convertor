
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
from Models import cnn_model
from dataset import SymbolDataset
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
    # Load all images as dataset.py (without augmentation)
    dataset = SymbolDataset("Data/Dataset/Font_letters", transform=torchvision.transforms.ToTensor())


    # plt.imshow((dataset[29][0].reshape((47, 27))), cmap='Greys')
    # plt.show()


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