"""
    This program could convert any image to
        it's corresponding Ascii-Art using
        convolutional neural network (as symbol
        classifier).
"""

# Import all libraries
import numpy as np
import torchvision
from tqdm import tqdm
import torch
import Functional
from data_loaders import ImgLoader
from Models import cnn_model
from datasets import AugmentedDataset
import matplotlib.pyplot as plt
import time

# Load image
img = Functional.load_image("Data/Dataset/Font.png")
depth, high, width = img.shape
img = img.reshape(high, width)

# Configure model parameters
# used_symbols = [...]
mode = "train"
model_data = torch.load("Data/Model_2e.pth")




device = Functional.gpu_check()
model = cnn_model(33).to(device)

if mode == "train":

    dataset = AugmentedDataset(bg_obj_class="", transform=torchvision.transforms.ToTensor())

    # Show last image
    img = dataset[len(dataset) - 1][0]
    plt.imshow(img.reshape(img.shape[1:]), cmap='Greys')
    plt.show()

    batch_loader = ImgLoader(dataset=dataset, batch_size=50, device=device)
    train_batch_loader, valid_batch_loader = batch_loader.get_loaders(split_ratio=0.9)

    model.train(mode=True)
    epochs = 3
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []

    for epoch in range(0, epochs):
        np.random.seed(44)

        for i, (img, ans) in enumerate(tqdm(train_batch_loader)):  # 0.30s
            ans = torch.nn.functional.one_hot(ans, num_classes=33)
            Y_pred = model.forward(img.view(-1, *img.shape[1:]))  # 0.0005s
            ans = ans.to(torch.float32)

            l = loss_func(Y_pred, ans)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 5 == 0:
                history.append(l.data)

        plt.plot(list(map(lambda x: x.cpu(), history)))
        plt.show()


        # Save model in each epoch
        parameters = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "epoch": epoch,
                      "history": history}

        torch.save(parameters, f"Data/Model_{epoch}e.pth")

if mode == "test":
    plt.imshow(img.reshape(320, 609), cmap='Greys')
    plt.show()

    model.load_state_dict(model_data["model"])
    model.eval()

    with torch.no_grad():
        pass

# start_time = time.time()
# print(time.time() - start_time)
