import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
class ESPCN(nn.Module):
    def __init__(self, scale_factor):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, scale_factor**2 * 3, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x
