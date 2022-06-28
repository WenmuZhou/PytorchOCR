# -*- coding: utf-8 -*-
"""
@time: 2021/2/8 21:28
@author: Bourne-M
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torchocr.networks.CommonModules import  SEBlock




class DSConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=1,
                 groups=None,
                 if_act=True,
                 act="relu",
                 **kwargs):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias=False)

        self.bn2 = nn.BatchNorm2d(in_channels * 4)

        self.conv3 = nn.Conv2d(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False)

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x

def weights_init(m):

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


class DB_fpn(nn.Module):
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
        self.p5_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(self.out_channels, self.out_channels // 4, kernel_size=3, padding=1, bias=False)

        self.in2_conv.apply(weights_init)
        self.in3_conv.apply(weights_init)
        self.in4_conv.apply(weights_init)
        self.in5_conv.apply(weights_init)
        self.p5_conv.apply(weights_init)
        self.p4_conv.apply(weights_init)
        self.p3_conv.apply(weights_init)
        self.p2_conv.apply(weights_init)

    def _interpolate_add(self, x, y):
        return F.interpolate(x, scale_factor=2) + y

    def _interpolate_cat(self, p2, p3, p4, p5):
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

        out4 = self._interpolate_add(in5, in4)
        out3 = self._interpolate_add(out4, in3)
        out2 = self._interpolate_add(out3, in2)

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        x = self._interpolate_cat(p2, p3, p4, p5)
        return x

class RSELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False)
        self.se_block = SEBlock(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut))

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        # out4 = in4 + F.interpolate(
        #     in5, scale_factor=2, mode="nearest")  # 1/16
        # out3 = in3 + F.interpolate(
        #     out4, scale_factor=2, mode="nearest")  # 1/8
        # out2 = in2 + F.interpolate(
        #     out3, scale_factor=2, mode="nearest")  # 1/4 
        out4 = in4 + F.interpolate(in5, scale_factor=2) 
        out3 = in3 + F.interpolate(
            out4, scale_factor=2)  # 1/8
        out2 = in2 + F.interpolate(
            out3, scale_factor=2)  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class LKPAN(nn.Module):
    def __init__(self, in_channels, out_channels, mode='large', **kwargs):
        super(LKPAN, self).__init__()
        self.out_channels = out_channels

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        # pan head
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        if mode.lower() == 'lite':
            p_layer = DSConv
        elif mode.lower() == 'large':
            p_layer = nn.Conv2d
        else:
            raise ValueError(
                "mode can only be one of ['lite', 'large'], but received {}".
                format(mode))

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    bias=False))

            self.inp_conv.append(
                p_layer(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias=False))

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False))
            self.pan_lat_conv.append(
                p_layer(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    bias=False))

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(
            in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(
            out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(
            out3, scale_factor=2, mode="nearest")  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse