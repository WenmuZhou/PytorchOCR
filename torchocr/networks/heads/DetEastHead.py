from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


class EastHead(nn.Module):
    def __init__(self, in_channels, scope = 800):
        super(EastHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, 8, 1)
        self.scope = scope

    def load_3rd_state_dict(self, _3rd_name, _state):
        pass

    def forward(self, x):
        score = self.conv1(x)
        score = F.sigmoid(score)
        geo = self.conv2(x)
        geo = (F.sigmoid(geo) - 0.5) * 2 * self.scope
        return score, geo
