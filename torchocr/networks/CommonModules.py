import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HardSigmoid(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type

    def forward(self, x):
        if self.type == 'paddle':
            x = (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
        else:
            x = F.relu6(x + 3, inplace=True) / 6
        return x


class HSigmoid(nn.Module):
    def forward(self, x):
        x = (1.2 * x).add_(3.).clamp_(0., 6.).div_(6.)
        return x


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hsigmoid_type='others', ratio=4):
        super().__init__()
        num_mid_filter = out_channels // ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, kernel_size=1, out_channels=out_channels, bias=True)
        self.relu2 = HardSigmoid(hsigmoid_type)

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4):
        super(ChannelAttention, self).__init__()
        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.__fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                                  nn.ReLU(True),
                                  nn.Conv2d(in_channels // ratio, out_channels, 1, bias=False), )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)
        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)
        y = self.__sigmoid(y1 + y2)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4):
        super(CBAM, self).__init__()
        self.cam = ChannelAttention(in_channels, out_channels, ratio)
        self.sam = SpatialAttention()

    def forward(self, x):
        x = self.cam(x)
        x = x*self.sam(x)
        return x
