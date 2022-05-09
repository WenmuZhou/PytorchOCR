# -*- coding: utf-8 -*-
"""
@time: 2021/2/8 21:28
@author: Bourne-M
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchocr.networks.CommonModules import ScaleFeatureSelection
import numpy as np


def weights_init(m):
    import torch.nn.init as init
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)


def weights_init(m):
    import torch.nn.init as init
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)


class DB_Asf(nn.Module):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.out_channels = out_channels
        # reduce layers
        self.in2_conv = nn.Conv2d(in_channels[0], self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels[1], self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels[2], self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels[3], self.out_channels, kernel_size=1, bias=False)
        # Smooth layers
        self.p5_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=8, mode='nearest'))

        self.p4_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.p3_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.p2_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)

        self.concat_attention = ScaleFeatureSelection(out_channels, out_channels // 4,
                                                      attention_type='scale_channel_spatial')

        self.in2_conv.apply(weights_init)
        self.in3_conv.apply(weights_init)
        self.in4_conv.apply(weights_init)
        self.in5_conv.apply(weights_init)
        self.p5_conv.apply(weights_init)
        self.p4_conv.apply(weights_init)
        self.p3_conv.apply(weights_init)
        self.p2_conv.apply(weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        p3 = F.interpolate(p3, scale_factor=2)
        p4 = F.interpolate(p4, scale_factor=4)
        p5 = F.interpolate(p5, scale_factor=8)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = self._upsample_add(in5, in4)
        out3 = self._upsample_add(out4, in3)
        out2 = self._upsample_add(out3, in2)

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.concat_attention(fuse, [p5, p4, p3, p2])

        return fuse
