import importlib
import copy

from .rec_metric import RecMetric
from .det_metric import DetMetric
from .e2e_metric import E2EMetric
from .cls_metric import ClsMetric


class DistillationMetric(object):
    def __init__(self,
                 key=None,
                 base_metric_name=None,
                 main_indicator=None,
                 **kwargs):
        self.main_indicator = main_indicator
        self.key = key
        self.main_indicator = main_indicator
        self.base_metric_name = base_metric_name
        self.kwargs = kwargs
        self.metrics = None

    def _init_metrcis(self, preds):
        self.metrics = dict()
        mod = importlib.import_module(__name__)
        for key in preds:
            self.metrics[key] = getattr(mod, self.base_metric_name)(
                main_indicator=self.main_indicator, **self.kwargs)
            self.metrics[key].reset()

    def __call__(self, preds, batch, **kwargs):
        assert isinstance(preds, dict)
        if self.metrics is None:
            self._init_metrcis(preds)
        output = dict()
        for key in preds:
            self.metrics[key].__call__(preds[key], batch, **kwargs)

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        output = dict()
        for key in self.metrics:
            metric = self.metrics[key].get_metric()
            # main indicator
            if key == self.key:
                output.update(metric)
            else:
                for sub_key in metric:
                    output["{}_{}".format(key, sub_key)] = metric[sub_key]
        return output

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()
