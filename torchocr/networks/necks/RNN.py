# encoding: utf-8
"""
@time: 2021/2/9 22:21
@author: Bourne-M
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels,**kwargs):
        super(EncoderWithRNN, self).__init__()
        hidden_size = kwargs.get('hidden_size', 256)
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, num_layers=2,batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type='rnn',  **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'rnn': EncoderWithRNN
            }
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())

            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels,**kwargs)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        x = self.encoder_reshape(x)
        if not self.only_reshape:
            x = self.encoder(x)

        return x



