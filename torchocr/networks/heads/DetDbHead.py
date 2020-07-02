from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def load_3rd_state_dict(self, _3rd_name, _state):
        pass

    def forward(self, x):
        return self.out(x)


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
            return shrink_maps.detach().cpu().numpy()
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
