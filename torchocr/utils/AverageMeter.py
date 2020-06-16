# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 14:28
# @Author  : zhoujun

from collections import defaultdict, deque


class AverageMeter:
    """
    用于方便进行值的统计
    """

    def __init__(self):
        self.value_accumulate = defaultdict(float)
        self.total_count = defaultdict(int)

    def __setitem__(self, key, value):
        assert len(value) == 2
        self.value_accumulate[key] += value[0]
        self.total_count[key] += value[1]

    def __getitem__(self, key):
        return self.value_accumulate[key] / self.total_count[key]
