# -*- coding:utf-8 -*-
# @Time    : 2020/6/21 11:40
# @author :lgc
import torch.nn.functional as F
from torch import nn, Tensor
import torch


class EastFeatureFusion(nn.Module):

    def __init__(self, in_channels, out_channels=[32, 32, 64, 128]):
        super(EastFeatureFusion, self).__init__()
        self.out_channels = out_channels
        
        self.merge1 = nn.Sequential(
                         nn.Conv2d(in_channels[3] + in_channels[2], out_channels[3], 1),
                         nn.BatchNorm2d(out_channels[3]),
                         nn.ReLU(),
                         nn.Conv2d(out_channels[3], out_channels[3], 3, padding=1),
                         nn.BatchNorm2d(out_channels[3]),
                         nn.ReLU()
                         )
        
        self.merge2 = nn.Sequential(
                      nn.Conv2d(in_channels[1] + out_channels[3], out_channels[2], 1),
                      nn.BatchNorm2d(out_channels[2]),
                      nn.ReLU(),
                      nn.Conv2d(out_channels[2], out_channels[2], 3, padding=1),
                      nn.BatchNorm2d(out_channels[2]),
                      nn.ReLU(),                     
                      )
        
        self.merge3 = nn.Sequential(
                      nn.Conv2d(in_channels[0] + out_channels[2], out_channels[1], 1),
                      nn.BatchNorm2d(out_channels[1]),
                      nn.ReLU(),
                      nn.Conv2d(out_channels[1], out_channels[1], 3, padding=1),
                      nn.BatchNorm2d(out_channels[1]),
                      nn.ReLU()                      
                      )
        
        self.merge4 = nn.Sequential(
                      nn.Conv2d(out_channels[0], out_channels[0], 3, padding=1),
                      nn.BatchNorm2d(out_channels[0]),
                      nn.ReLU()                      
                      )
        
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = F.interpolate(x[3], scale_factor=2, mode='bilinear')
        f = torch.cat((f, x[2]), 1)
        f = self.merge1(f)
        
        f = F.interpolate(f, scale_factor=2, mode='bilinear')
        f = torch.cat((f, x[1]), 1)
        f = self.merge2(f)
        
        f = F.interpolate(f, scale_factor=2, mode='bilinear')
        f = torch.cat((f, x[0]), 1)
        f = self.merge3(f)
        
        f = self.merge4(f)        

        return f


class EastFeatureFusion_Paddle(nn.Module):
    def forward(self, results, x, names):
        pass


