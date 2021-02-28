from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

from torchocr.networks.CommonModules import ConvBNACT, SEBlock


class ResidualUnit(nn.Module):
    def __init__(self, num_in_filter, num_mid_filter, num_out_filter, stride, kernel_size, act=None, use_se=False):
        super().__init__()
        self.expand_conv = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                               padding=0, act=act)

        self.bottleneck_conv = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter, kernel_size=kernel_size,
                               stride=stride,
                               padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter, out_channels=num_mid_filter,hsigmoid_type='paddle')
        else:
            self.se = None

        self.linear_conv = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                               padding=0)
        self.not_add = num_in_filter != num_out_filter or stride != 1

    def load_3rd_state_dict(self, _3rd_name, _state, _convolution_index):
        if _3rd_name == 'paddle':
            self.expand_conv.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_expand')
            self.bottleneck_conv.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_depthwise')
            if self.se is not None:
                self.se.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_se')
            self.linear_conv.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_linear')
        else:
            pass
        pass

    def forward(self, x):
        y = self.expand_conv(x)
        y = self.bottleneck_conv(y)
        if self.se is not None:
            y = self.se(y)
        y = self.linear_conv(y)
        if not self.not_add:
            y = x + y
        return y

class MobileNetV3(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.scale = kwargs.get('scale', 0.5)
        model_name = kwargs.get('model_name', 'small')
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', (2, 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (2, 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 1],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', (2, 1)],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (1, 1)],
                [3, 72, 24, False, 'relu', (2, 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', (2, 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', (2, 1)],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, "supported scale are {} but input scale is {}".format(supported_scale,
                                                                                                    self.scale)

        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        # conv1
        self.conv1 = ConvBNACT(in_channels=in_channels,
                               out_channels=self.make_divisible(inplanes * scale),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               act='hard_swish')
        inplanes = self.make_divisible(inplanes * scale)
        block_list = []
        for layer_cfg in cfg:
            block = ResidualUnit(num_in_filter=inplanes,
                                 num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                                 num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                                 act=layer_cfg[4],
                                 stride=layer_cfg[5],
                                 kernel_size=layer_cfg[0],
                                 use_se=layer_cfg[3])
            block_list.append(block)
            inplanes = self.make_divisible(scale * layer_cfg[2])

        self.blocks = nn.Sequential(*block_list)
        self.conv2 = ConvBNACT(in_channels=inplanes,
                               out_channels=self.make_divisible(scale * cls_ch_squeeze),
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               act='hard_swish')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = self.make_divisible(scale * cls_ch_squeeze)

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def load_3rd_state_dict(self, _3rd_name, _state):
        if _3rd_name == 'paddle':
            self.conv1.load_3rd_state_dict(_3rd_name, _state, 'conv1')
            for m_block_index, m_block in enumerate(self.blocks, 2):
                m_block.load_3rd_state_dict(_3rd_name, _state, m_block_index)
            self.conv2.load_3rd_state_dict(_3rd_name, _state, 'conv_last')
        else:
            pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x