# -*- coding: utf-8 -*-
# @Time    : 2020/6/21 13:15
# @Author  : lgc
from addict import Dict as AttrDict
from torch import nn

from torchocr.networks.backbones.DetMobilenetV3 import MobileNetV3
from torchocr.networks.backbones.DetResNetvd import ResNet
from torchocr.networks.necks.EastFeatureFusion import EastFeatureFusion
from torchocr.networks.heads.DetEastHead import EastHead

backbone_dict = {'MobileNetV3': MobileNetV3, 'ResNet': ResNet}
neck_dict = {'FPN': EastFeatureFusion}
head_dict = {'EastHead': EastHead}


class DetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert 'in_channels' in config, 'in_channels must in model config'
        backbone_type = config.backbone.pop('type')
        assert backbone_type in backbone_dict, f'backbone.type must in {backbone_dict}'
        self.backbone = backbone_dict[backbone_type](config.in_channels, **config.backbone)

        neck_type = config.neck.pop('type')
        assert neck_type in neck_dict, f'neck.type must in {neck_dict}'
        self.neck = neck_dict[neck_type](self.backbone.out_channels, **config.neck)
        head_type = config.head.pop('type')
        assert head_type in head_dict, f'head.type must in {head_dict}'
        self.head = head_dict[head_type](self.neck.out_channels[0], **config.head)
        

        self.name = f'DetModel_{backbone_type}_{neck_type}_{head_type}'

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.backbone.load_3rd_state_dict(_3rd_name, _state)
        self.neck.load_3rd_state_dict(_3rd_name, _state)
        self.head.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        score, geo = self.head(x)
        return score, geo


if __name__ == '__main__':
    # from torchocr.model_config import AttrDict
    import torch

    East_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV3', scale=0.5, model_name='large'),
        neck=AttrDict(type='FPN', out_channels=[32, 32, 64, 128]),
        head=AttrDict(type='EastHead')
    )
    x = torch.zeros(1, 3, 512, 512)
    model = DetModel(East_config)
    #print(model)
    y = model(x)
    print(model.name)
    print(y[0].shape)
    print(y[1].shape)