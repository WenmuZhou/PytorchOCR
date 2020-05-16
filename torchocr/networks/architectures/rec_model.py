# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 11:18
# @Author  : zhoujun
from torch import nn

from torchocr.networks.backbones.rec_mobilenet_v3 import MobileNetV3
from torchocr.networks.necks.rec_sequence_decoder import SequenceDecoder
from torchocr.networks.heads.rec_ctc_head import CTC


class RECModel(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.backbone = MobileNetV3(in_channels)
        d = {'type': 'rnn'}
        self.neck = SequenceDecoder(480, **d)
        self.head = CTC(384, 37)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import torch
    x = torch.zeros(1,3,32,320)
    model = RECModel(3)
    y= model(x)
    print(y.shape)
    torch.save(model.state_dict(),'1,pth')