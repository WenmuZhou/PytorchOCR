from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1,
                               bias=False)
        self.conv_bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=2,
                                        stride=2)
        self.conv_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=2, stride=2)

    def load_3rd_state_dict(self, _3rd_name, _state):
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = Head(in_channels)
        self.thresh = Head(in_channels)
        self.binarize.apply(self.weights_init)
        self.thresh.apply(self.weights_init)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def load_3rd_state_dict(self, _3rd_name, _state):
        pass

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return shrink_maps
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return y

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
