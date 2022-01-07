from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch.nn.functional as F
from torchocr.networks.CommonModules import ConvBNACT


class PseHead(nn.Module):
    def __init__(self, in_channels, result_num=6, **kwargs):
        super(PseHead, self).__init__()
        self.H = kwargs.get('H', 640)
        self.W = kwargs.get('W', 640)
        self.scale = kwargs.get('scale', 1)
        self.conv = ConvBNACT(in_channels, in_channels // 4, kernel_size=3, padding=1, stride=1, act='relu')
        self.out_conv = nn.Conv2d(in_channels // 4, result_num, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.out_conv(x)
        if self.train:
            x = F.interpolate(x, size=(self.H, self.W), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=(self.H // self.scale, self.W // self.scale), mode='bilinear', align_corners=True)
        return x
