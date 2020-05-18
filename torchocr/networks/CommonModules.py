import torch
from torch import nn
from torch.nn import functional as F


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HardSigmoid(nn.Module):
    def __init__(self, slope=.2, offset=.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        x = (self.slope * x) + self.offset
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x
