"""
    This program could convert any image to
        it's corresponding Ascii-Art using
        convolutional neural network (as symbol
        classifier).
"""

# Import all libraries
import random
import numpy as np
import torchvision
from tqdm import tqdm
import torch
import Functional
from data_loaders import ImgLoader
import Models
from datasets import AugmentedDataset
import matplotlib.pyplot as plt
import time

# Load image
img = Functional.load_image("Data/Dataset/Test.jpg")
img = torchvision.transforms.Grayscale()(img)
depth, high, width = img.shape
img = img.reshape(high, width)

# Configure model parameters
used_symbols = "<)«¬-¯/+-(»‹›_¤+>EFHKLOT}{|~^]_\["
magic_number = 69
mode = "train"
model_data = torch.load("Data/Model_Configs/Basic_CNN_2e.pth")




device = Functional.gpu_check()
model = Models.cnn_model_v2(33).to(device)

if mode == "train":

    dataset = AugmentedDataset(bg_obj_class="", transform=torchvision.transforms.ToTensor())

    # Show last image
    img = dataset[len(dataset) - 1][0]
    plt.imshow(img.reshape(img.shape[1:]), cmap='Greys')
    plt.show()

    batch_loader = ImgLoader(dataset=dataset, batch_size=50, device=device)
    train_batch_loader, valid_batch_loader = batch_loader.get_loaders(split_ratio=0.9)

    model.train()
    epochs = 3
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []

    for epoch in range(0, epochs):
        torch.manual_seed(magic_number)
        np.random.seed(magic_number)
        random.seed(magic_number)

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

        torch.save(parameters, f"Data/Model_Configs/{model.name}_{epoch}e.pth")

if mode == "test":
    # print(summary(model, (1, 47, 27), batch_size=50))
    plt.imshow(img, cmap='Greys')
    plt.show()

    model.load_state_dict(model_data["model"])
    model.eval()

    with torch.no_grad():
        out_art = ""

        for i in range(0, high // 47):
            out_art += "\n"
            for j in range(0, width // 27):
                img_segment = img[i*47:(i+1)*47, j*27:(j+1)*27]
                img_segment = img_segment.reshape(1, 47, 27).to(device)
                pred = model.forward(img_segment)
                pred = torch.nn.Sigmoid()(pred).argmax()
                out_art += used_symbols[pred]

        print(out_art)
                
            

# start_time = time.time()
# print(time.time() - start_time)
