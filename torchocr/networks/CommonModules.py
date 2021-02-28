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

    def load_3rd_state_dict(self, _3rd_name, _state, _name_prefix):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            to_load_state_dict['conv.weight'] = torch.Tensor(_state[f'{_name_prefix}_weights'])
            to_load_state_dict['bn.weight'] = torch.Tensor(_state[f'{_name_prefix}_bn_scale'])
            to_load_state_dict['bn.bias'] = torch.Tensor(_state[f'{_name_prefix}_bn_offset'])
            to_load_state_dict['bn.running_mean'] = torch.Tensor(_state[f'{_name_prefix}_bn_mean'])
            to_load_state_dict['bn.running_var'] = torch.Tensor(_state[f'{_name_prefix}_bn_variance'])
            self.load_state_dict(to_load_state_dict)
        else:
            pass

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

    def load_3rd_state_dict(self, _3rd_name, _state, _name_prefix):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            to_load_state_dict['conv1.weight'] = torch.Tensor(_state[f'{_name_prefix}_1_weights'])
            to_load_state_dict['conv2.weight'] = torch.Tensor(_state[f'{_name_prefix}_2_weights'])
            to_load_state_dict['conv1.bias'] = torch.Tensor(_state[f'{_name_prefix}_1_offset'])
            to_load_state_dict['conv2.bias'] = torch.Tensor(_state[f'{_name_prefix}_2_offset'])
            self.load_state_dict(to_load_state_dict)
        else:
            pass

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn
