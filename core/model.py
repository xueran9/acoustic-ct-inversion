"""AI反演神经网络模型"""
import torch
import torch.nn as nn


class Net(nn.Module):
    """用于声波CT反演的神经网络"""
    def __init__(self, size=64):
        super(Net, self).__init__()
        self.size = size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(90 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, size * size),
            nn.Unflatten(1, (size, size))
        )

    def forward(self, x):
        x = self.fc(x)
        return 1400 + 200 * torch.sigmoid(x)
