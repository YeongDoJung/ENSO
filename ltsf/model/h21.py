import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.modules import padding

class Model2D(nn.Module):
    def __init__(self, num_conv=50, num_dense=50):
        super().__init__()
        inner_channel = 35
        inner_dense = 50
        self.conv1 = nn.Sequential(nn.Conv2d(6, inner_channel, kernel_size=(8,4), padding=(4,2)), nn.Tanh(), nn.MaxPool2d(kernel_size=(2,2)))
        self.conv2 = nn.Sequential(nn.Conv2d(inner_channel, inner_channel, kernel_size=(4,2), padding=(2,1)), nn.Tanh(), nn.MaxPool2d(kernel_size=(2,2)))
        self.conv3 = nn.Sequential(nn.Conv2d(inner_channel, inner_channel, kernel_size=(4,2)), nn.Tanh())
        self.dense1 = nn.Sequential(nn.Linear(inner_channel, inner_dense), nn.Tanh())
        self.dense2 = nn.Sequential(nn.Linear(inner_dense, inner_dense), nn.Tanh())
        self.dense3 = nn.Linear(inner_dense, 35)
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        ninoprediction = x[:, :23]
        season = self.softmax(x[:, 23:])
        
        return ninoprediction, season

