import torch.nn as nn



class cnn_model(nn.Module):
    def __init__(self, out_class_num):
        super(cnn_model, self).__init__()
        self.name = "Basic_CNN"
        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 10, 3, stride=1),
            nn.ReLU(),

            nn.Conv2d(10, 30, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(30, 50, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(50, 100, 2, stride=1),
            nn.ReLU(),

            nn.Conv2d(100, 150, 2, stride=1),
            nn.ReLU(),
        )

        self.lin_model = nn.Sequential(
            nn.Linear(150 * 7 * 2, 500),
            nn.ReLU(),

            nn.Linear(500, 200),
            nn.ReLU(),

            nn.Linear(200, out_class_num)
        )

    def forward(self, X):
        out = self.conv_model(X)
        out = out.view(-1, 150 * 7 * 2)
        out = self.lin_model(out)
        return out






# This one is quite similar to ResNet
class cnn_model_v2(nn.Module):

    def __init__(self, out_class_num):
        super(cnn_model_v2, self).__init__()
        self.name = "ResCNN_v1"
        self.block_1 = nn.Sequential(
            self.conv_builder(1, 10, 1),
            self.conv_builder(10, 30, 3, padding=1),
            self.conv_builder(30, 50, 3, padding=1),
            nn.MaxPool2d(2)
        )

        self.downsample_1 = self.downsample(1, 50, maxpool=2)

        self.block_2 = nn.Sequential(
            self.conv_builder(50, 100, 1),
            self.conv_builder(100, 150, 3, padding=1),
            self.conv_builder(150, 300, 3, padding=1),
            nn.MaxPool2d(2)
        )

        self.downsample_2 = self.downsample(1, 300, maxpool=4)

        self.block_3 = nn.Sequential(
            self.conv_builder(300, 500, 2, padding=1),
            self.conv_builder(500, 750, 2),
            nn.MaxPool2d(2)
        )

        self.downsample_3 = self.downsample(1, 750, maxpool=8)

        self.lin_model = nn.Sequential(
            nn.Linear(750*5*3, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 200),
            nn.ReLU(),

            nn.Linear(200, out_class_num)
        )

    def forward(self, X):
        out = self.block_1(X)
        out += self.downsample_1(X)     #Skip connection
        out = self.block_2(out)
        out += self.downsample_2(X)     #Skip connection
        out = self.block_3(out)
        out += self.downsample_3(X)     #Skip connection
        out = out.view(-1, 750*5*3)
        out = self.lin_model(out)
        return out



    def conv_builder(self, input_dim, output_dim, kernel, stride=1, padding=0):
        conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
        return conv

    def downsample(self, input_dim, output_dim, maxpool=0):
        downsample = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, stride=1),
            nn.BatchNorm2d(output_dim),
            nn.MaxPool2d(maxpool)
        )
        return downsample

