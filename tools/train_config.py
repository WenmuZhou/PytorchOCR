# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing


# for train
resume_from = None
ckpt_dir = ""
use_cuda = True
device = 'cuda:0'
train_options = {
        'base_lr': 1.0,
        'batch_size': 64,
        'epochs': 200,
        'weight_decay': 1e-8,
}

SEED = 927
# if autoscale_lr:
#     # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
#     optimizer['lr'] = optimizer['lr'] * len(gpu_ids) / 8
# autoscale_lr = True

# for model
architecture = 'CRNNRes34'
architecture_config = {
    'in_channels': 3,
    'labels': 1000
}


# for dataset


# 其他
information_verbose = False
use_loguru = False
if use_loguru:
    from loguru import logger
    import sys

    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
else:
    # 自用简易logger
    from functools import partial
    import termcolor


    class Logger:
        def __init__(self):
            keywords = ['debug', 'info', 'error']
            background_colors = ['yellow', 'grey', 'white']
            foreground_colors = ['on_red', 'on_green', 'on_grey']
            for m_keyword, m_background_color, m_foreground in zip(keywords, background_colors, foreground_colors):
                self.__setattr__(m_keyword, partial(termcolor.cprint, color=m_background_color, on_color=m_foreground))


    logger = Logger()