# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/ppdet/modeling/necks/fpn.py
"""

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['FCEFPN']

class FCEFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_c5=True,
                 ):
        super(FCEFPN, self).__init__()
        self.out_channels = out_channels
        self.use_c5 = use_c5
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs =nn.ModuleList()
        # stage index 0,1,2,3 stands for res2,res3,res4,res5 on ResNet Backbone
        # 0 <= st_stage < ed_stage <= 3
        st_stage = 4 - len(in_channels)
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            in_c = in_channels[i - st_stage]
            self.lateral_convs.append( nn.Conv2d(
                    in_channels=in_c,
                    out_channels=out_channels,
                    kernel_size=1))

        for i in range(st_stage, ed_stage + 1):
            self.fpn_convs.append(nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1))

        # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
    def forward(self, body_feats):
        laterals = []
        num_levels = len(body_feats)

        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest')
            laterals[lvl - 1] += upsample

        fpn_output = []
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        return fpn_output
