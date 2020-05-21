# -*- coding:utf-8 -*-
# @author :adolf
import torch
from torch import nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        assert len(in_channels) == 4
        self.out_channels = out_channels
        self.reduce_c5 = nn.Conv2d(in_channels=in_channels[3], out_channels=self.out_channels, kernel_size=1,
                                   bias=False)
        self.reduce_c4 = nn.Conv2d(in_channels=in_channels[2], out_channels=self.out_channels, kernel_size=1,
                                   bias=False)
        self.reduce_c3 = nn.Conv2d(in_channels=in_channels[1], out_channels=self.out_channels, kernel_size=1,
                                   bias=False)
        self.reduce_c2 = nn.Conv2d(in_channels=in_channels[0], out_channels=self.out_channels, kernel_size=1,
                                   bias=False)

        self.smooth_p5 = nn.Conv2d(in_channels=out_channels, out_channels=self.out_channels // 4, kernel_size=3,
                                   padding=1, bias=False)
        self.smooth_p4 = nn.Conv2d(in_channels=out_channels, out_channels=self.out_channels // 4, kernel_size=3,
                                   padding=1, bias=False)
        self.smooth_p3 = nn.Conv2d(in_channels=out_channels, out_channels=self.out_channels // 4, kernel_size=3,
                                   padding=1, bias=False)
        self.smooth_p2 = nn.Conv2d(in_channels=out_channels, out_channels=self.out_channels // 4, kernel_size=3,
                                   padding=1, bias=False)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.reduce_c5(c5)
        in4 = self.reduce_c4(c4)
        in3 = self.reduce_c3(c3)
        in2 = self.reduce_c2(c2)

        out4 = self._upsample_add(in5, in4)
        out3 = self._upsample_add(out4, in3)
        out2 = self._upsample_add(out3, in2)

        p5 = self.smooth_p5(in5)
        p4 = self.smooth_p4(out4)
        p3 = self.smooth_p3(out3)
        p2 = self.smooth_p2(out2)

        out = self._upsample_cat(p2, p3, p4, p5)
        return out

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)


if __name__ == '__main__':
    a = torch.zeros([1, 64, 160, 160])

    b = torch.zeros([1, 128, 80, 80])

    c = torch.zeros([1, 256, 40, 40])

    d = torch.zeros([1, 512, 20, 20])

    inputs = [a, b, c, d]
    fpn = FPN(in_channels=[64, 128, 256, 512])
    y = fpn(inputs)
    print(y.size())
