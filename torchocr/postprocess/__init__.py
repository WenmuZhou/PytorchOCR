# -*- coding: utf-8 -*-
# @Time    : 2020/5/15 17:41
# @Author  : zhoujun

import copy
from .DBPostProcess import DBPostProcess

support_post_process = ['DBPostProcess', 'DetModel']


def build_post_process(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    post_process_type = copy_config.pop('type')
    assert post_process_type in support_post_process, f'{post_process_type} is not developed yet!, only {support_post_process} are support now'
    post_process = eval(post_process_type)(**copy_config)
    return post_process
