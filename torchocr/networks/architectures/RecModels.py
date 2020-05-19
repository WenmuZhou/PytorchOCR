# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 11:18
# @Author  : zhoujun
from torch import nn

from torchocr.networks.backbones.RecMobileNetV3 import MobileNetV3
from torchocr.networks.backbones.RecResNet34vd import ResNet
from torchocr.networks.heads.RecCTCHead import CTC
from torchocr.networks.necks.RecSequenceDecoder import SequenceDecoder


class CRNNMBV3(nn.Module):
    def __init__(self, in_channels, labels, **kwargs):
        super().__init__()
        self.backbone = MobileNetV3(in_channels, scale=0.5, model_name='large')
        self.neck = SequenceDecoder(480, type='rnn', hidden_size=96)
        self.head = CTC(192, labels)

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.backbone.load_3rd_state_dict(_3rd_name, _state)
        self.neck.load_3rd_state_dict(_3rd_name, _state)
        self.head.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


class CRNNRes34(nn.Module):
    def __init__(self, in_channels, labels, **kwargs):
        super().__init__()
        self.backbone = ResNet(in_channels, 34)
        self.neck = SequenceDecoder(512, type='rnn', hidden_size=256)
        self.head = CTC(512, labels)

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.backbone.load_3rd_state_dict(_3rd_name, _state)
        self.neck.load_3rd_state_dict(_3rd_name, _state)
        self.head.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

