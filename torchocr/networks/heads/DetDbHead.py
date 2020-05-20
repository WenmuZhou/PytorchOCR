from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import paddle.fluid as fluid
from torch import nn


class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.out(x)


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = Head(in_channels)
        self.thresh = Head(in_channels)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return shrink_maps.detach().cpu().numpy()
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return y

if __name__ == '__main__':
    x = torch.zeros(1,256,160,160)
    model = DBHead(256)
    model.eval()
    y = model(x)
    print(y.shape)
    from torchocr.postprocess.DBPostProcess import DBPostProcess
    p = DBPostProcess()
    b,c = p(y,[(640,640)])