# -*- coding: utf-8 -*-
# @Time    : 2020/5/15 17:43
# @Author  : zhoujun
import copy
from addict import Dict
from .DBLoss import DBLoss
from .CTCLoss import CTCLoss
from .PSELoss import PSELoss
from .CombinedLoss import CombinedLoss

__all__ = ['build_loss']

support_loss = ['DBLoss', 'CTCLoss','PSELoss','CombinedLoss']


def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    # criterion = eval(loss_type)(**copy_config)
    criterion = eval(loss_type)(Dict(copy_config))
    return criterion
