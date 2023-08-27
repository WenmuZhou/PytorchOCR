import copy

__all__ = ["build_metric"]

from .det_metric import DetMetric, DetFCEMetric
from .rec_metric import RecMetric, CNTMetric, CANMetric
from .cls_metric import ClsMetric
from .e2e_metric import E2EMetric
from .distillation_metric import DistillationMetric


def build_metric(config):
    support_dict = [
        "DetMetric", "DetFCEMetric", "RecMetric", "ClsMetric", "E2EMetric",
        "DistillationMetric"
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
