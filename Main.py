
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
from data_loaders import ImgLoader
from Models import cnn_model
from datasets import AugmentedDataset
import matplotlib.pyplot as plt

# # Load image
# img = ...
#
# # Configure global variables
# used_symbols = [...]
mode = "train"
# width, hight, depth = img.shape
# img_shape = [width, hight, depth]




device = checks.gpu_check()
model = cnn_model().to()

if mode == "train":

    dataset = AugmentedDataset(bg_obj_class="")
    img = dataset[len(dataset) - 1][0]
    plt.imshow(img, cmap='Greys')
    plt.show()

    batch_loader = ImgLoader(dataset=dataset, batch_size=50, device=device)
    train_batch_loader, valid_batch_loader = batch_loader.get_loaders(split_ratio=0.9)


    model.train(mode=True)
    epochs = 2
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    history = []

    for epoch in range(0, epochs):
        np.random.seed(44)

        for i, (img, ans) in enumerate(tqdm(train_batch_loader)):
            Y_pred = model.forward(img.view(-1, torch.prod(img.shape)))
            l = loss_func(Y_pred, ans)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 3 == 0:
                history.append(l.data)



if mode == "test":
    model.train(mode=False)

    with torch.no_grad():
        pass