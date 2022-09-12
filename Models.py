import torch.nn as nn



class cnn_model(nn.Module):
    def __init__(self, out_class_num):
        super(cnn_model, self).__init__()
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








# class cnn_model(nn.Module):
#     def __init__(self, out_class_num):
#         super(cnn_model, self).__init__()
#
#         self.block_1 = nn.Sequential(
#             nn.Conv2d(1, 10, 3, stride=1),
#             nn.ReLU(),
#
#             nn.Conv2d(10, 30, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(30, 50, 3, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.block_2 = nn.Sequential(
#             nn.Conv2d(50, 100, 2, stride=1),
#             nn.ReLU(),
#
#             nn.Conv2d(100, 150, 2, stride=1),
#             nn.ReLU()
#         )
#
#         self.lin_model = nn.Sequential(
#             nn.Linear(150 * 7 * 2, 500),
#             nn.ReLU(),
#
#             nn.Linear(500, 200),
#             nn.ReLU(),
#
#             nn.Linear(200, out_class_num)
#         )
#
#     def forward(self, X):
#         print(X.shape)
#         out = self.block_1(X)
#         print(out.shape)
#         out = self.block_2(out)
#         out = self.lin_model(out)
#         return out
#
