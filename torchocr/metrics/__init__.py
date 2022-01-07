# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 14:07
# @Author  : zhoujun
import copy
from .RecMetric import RecMetric
from .DetMetric import DetMetric
from .distill_metric import DistillationMetric


def build_metric(config):
    support_dict = ["DistillationMetric"]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
