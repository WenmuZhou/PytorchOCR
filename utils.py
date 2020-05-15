from collections import defaultdict, deque
from functools import partial

import termcolor
import torch.nn as nn
import torch.nn.init as init


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class AverageMeter:
    """
    用于方便进行值的统计
    """

    def __init__(self, last_value_count=100):
        self.value_accumulate = defaultdict(float)
        self.total_count = defaultdict(int)
        self.last_value_queue = dict()
        self.last_value_count = last_value_count

    def __setitem__(self, key, value):
        self.value_accumulate[key] += value
        self.total_count[key] += 1
        if key not in self.last_value_queue:
            self.last_value_queue[key] = deque(maxlen=self.last_value_count)
        self.last_value_queue[key].append(value)

    def show_result(self):
        to_return = []
        to_return.append('-' * 30)
        to_return.append(f'\titem_name\ttimes\ttotal_average\tlast_{self.last_value_count}_average')
        for m_key in self.total_count:
            to_return.append('\t%s\t%d\t%.4f\t%.4f' % (
                m_key,
                self.total_count[m_key],
                self.value_accumulate[m_key] / self.total_count[m_key],
                sum(self.last_value_queue[m_key]) / self.last_value_count
            ))
        to_return.append('-' * 30)
        return '\n'.join(to_return)

    def __repr__(self):
        print(self.show_result())

    def __str__(self):
        return self.show_result()

    def __len__(self):
        return len(self.value_accumulate)


# 自用简易logger
class Logger:
    def __init__(self):
        keywords = ['debug', 'info', 'error']
        background_colors = ['yellow', 'grey', 'white']
        foreground_colors = ['on_red', 'on_green', 'on_grey']
        for m_keyword, m_background_color, m_foreground in zip(keywords, background_colors, foreground_colors):
            self.__setattr__(m_keyword, partial(termcolor.cprint, color=m_background_color, on_color=m_foreground))


def init_logger():
    return Logger()
