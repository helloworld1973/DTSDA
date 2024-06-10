import torch.nn as nn


class ActNetwork(nn.Module):
    def __init__(self, conv1_in_channels, conv1_out_channels, conv2_out_channels, kernel_size_num, in_features_size):
        super(ActNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_in_channels, out_channels=conv1_out_channels, kernel_size=(1, kernel_size_num)),
            nn.BatchNorm2d(conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels,
                      kernel_size=(1, kernel_size_num)),
            nn.BatchNorm2d(conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.in_features = in_features_size

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = x.view(-1, self.in_features)
        return x
