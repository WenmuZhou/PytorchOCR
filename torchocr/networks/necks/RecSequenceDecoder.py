# -*- coding: utf-8 -*-
# @Time    : 2020/5/15 17:46
# @Author  : zhoujun

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from torch import nn
import torch


class DecoderWithRNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        in_channels = kwargs['in_channels']
        rnn_hidden_size = kwargs.get('hidden_size', 96)
        self.layers = 2
        self.lstm = nn.LSTM(in_channels, rnn_hidden_size, bidirectional=True, batch_first=True, num_layers=self.layers)

    def load_3rd_state_dict(self, _3rd_name, _state):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            for i in range(self.layers):
                # fc与ih对应
                to_load_state_dict[f'lstm.weight_ih_l{i}'] = torch.Tensor(_state[f'lstm_st{i + 1}_fc1_w'])
                to_load_state_dict[f'lstm.weight_ih_l{i}_reverse'] = torch.Tensor(_state[f'lstm_st{i + 1}_fc2_w'])
                to_load_state_dict[f'lstm.bias_ih_l{i}'] = torch.Tensor(_state[f'lstm_st{i + 1}_fc1_b'])
                to_load_state_dict[f'lstm.bias_ih_l{i}_reverse'] = torch.Tensor(_state[f'lstm_st{i + 1}_fc2_b'])
                # out与hh对应
                to_load_state_dict[f'lstm.weight_hh_l{i}'] = torch.Tensor(_state[f'lstm_st{i + 1}_out1_w'])
                to_load_state_dict[f'lstm.weight_hh_l{i}_reverse'] = torch.Tensor(_state[f'lstm_st{i + 1}_out2_w'])
                to_load_state_dict[f'lstm.bias_hh_l{i}'] = torch.Tensor(_state[f'lstm_st{i + 1}_out1_b'])
                to_load_state_dict[f'lstm.bias_hh_l{i}_reverse'] = torch.Tensor(_state[f'lstm_st{i + 1}_out2_b'])
        else:
            pass

    def forward(self, x):
        x = self.lstm(x)[0]
        return x


class SequenceDecoder(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        decoder_type = kwargs.get('type', 'rnn')
        decoder_dict = {'rnn': DecoderWithRNN, 'none': None}
        assert decoder_type in decoder_dict, "Unsupport encoder_type:%s" % decoder_type
        kwargs['in_channels'] = in_channels
        if decoder_type != 'none':
            self.decoder = decoder_dict[decoder_type](**kwargs)
        else:
            self.decoder = None

    def reshape(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channel)s
        return x

    def load_3rd_state_dict(self, _3rd_name, _state):
        if self.decoder:
            self.decoder.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        x = self.reshape(x)
        if self.decoder:
            return self.decoder(x)
        return x
