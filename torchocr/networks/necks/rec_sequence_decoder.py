# -*- coding: utf-8 -*-
# @Time    : 2020/5/15 17:46
# @Author  : zhoujun

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class DecoderWithReshape(object):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channel)s
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, fc_out=None):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        if fc_out is not None:
            self.fc = nn.Linear(hidden_size * 2, fc_out)
        else:
            self.fc = None

    def forward(self, x):
        x, _ = self.rnn(x)
        if self.fc is not None:
            x = self.fc(x)
        return x


class DecoderWithRNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        in_channels = kwargs['in_channels']
        rnn_hidden_size = kwargs.get('hidden_size', 96)
        self.lstm1 = BidirectionalLSTM(in_channels, rnn_hidden_size, fc_out=rnn_hidden_size * 4)

    def forward(self, x):
        x = x.squeeze(axis=2)
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channel)s
        x = self.lstm1(x)
        return x


class SequenceDecoder(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        encoder_type = kwargs.get('type', 'rnn')
        decoder_dict = {'rnn': DecoderWithRNN, 'None': DecoderWithReshape}
        assert encoder_type in decoder_dict, "Unsupport encoder_type:%s" % self.encoder_type
        kwargs['in_channels'] = in_channels
        self.decoder = decoder_dict[encoder_type](**kwargs)

    def forward(self, x):
        return self.decoder(x)


if __name__ == '__main__':
    import torch

    x = torch.zeros(1, 480, 1, 80)
    d = {'type': 'rnn'}
    model = SequenceDecoder(480, **d)
    y = model(x)
    print(y.shape)
