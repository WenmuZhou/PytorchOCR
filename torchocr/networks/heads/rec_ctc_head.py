
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

class CTC(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, n_class)

    def forward(self, x):
        return self.fc(x)