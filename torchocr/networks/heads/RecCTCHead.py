from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import torch
from torch import nn


class CTC(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def load_3rd_state_dict(self, _3rd_name, _state):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            if _state['ctc_fc_b_attr'].size == self.n_class:
                to_load_state_dict['fc.weight'] = torch.Tensor(_state['ctc_fc_w_attr'].T)
                to_load_state_dict['fc.bias'] = torch.Tensor(_state['ctc_fc_b_attr'])
                self.load_state_dict(to_load_state_dict)
        else:
            pass

    def forward(self, x):
        return self.fc(x)
