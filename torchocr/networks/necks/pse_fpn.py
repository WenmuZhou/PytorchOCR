# -*- coding: utf-8 -*-
"""
@time: 2021/2/8 21:28
@author: Bourne-M
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchocr.networks.CommonModules import ConvBNACT


class PSEFpn(nn.Module):
    def __init__(self, in_channels, out_channels=256, inplace=True, **kwargs):
        super().__init__()
        self.out_channels = out_channels * 4
        self.toplayer = ConvBNACT(in_channels=in_channels[3], out_channels=out_channels, kernel_size=1, stride=1, padding=0, act='relu')
        self.latlayer1 = ConvBNACT(in_channels=in_channels[2], out_channels=out_channels, kernel_size=1, stride=1, padding=0, act='relu')
        self.latlayer2 = ConvBNACT(in_channels=in_channels[1], out_channels=out_channels, kernel_size=1, stride=1, padding=0, act='relu')
        self.latlayer3 = ConvBNACT(in_channels=in_channels[0], out_channels=out_channels, kernel_size=1, stride=1, padding=0, act='relu')
        # Smooth layers
        self.smooth1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, act='relu')
        self.smooth2 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, act='relu')
        self.smooth3 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, act='relu')

    def forward(self, x):
        c2, c3, c4, c5 = x
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)
        x = self._upsample_cat(p2, p3, p4, p5)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)
