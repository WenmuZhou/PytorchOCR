import importlib
from .DetMetric import  DetMetric

class DistillationMetric(object):
    def __init__(self, key=None, base_metric_name=None, main_indicator=None, **kwargs):
        self.main_indicator = main_indicator
        self.key = key
        self.main_indicator = main_indicator
        self.base_metric_name = base_metric_name
        self.kwargs = kwargs
        self.metrics = None
        self.out = dict()

    def _init_metric(self, preds):
        self.metrics = dict()
        mod = importlib.import_module(__name__)
        for key in preds:
            self.metrics[key] = getattr(mod, self.base_metric_name)(**self.kwargs)

    def __call__(self,batch, preds, **kwargs):
        assert isinstance(preds, dict), f'preds should be dict,not {type(preds)}'
        if self.metrics is None:
            self._init_metric(preds)

        for key in preds:
            self.out.setdefault(key, []).append(self.metrics[key].__call__( batch,preds[key], **kwargs))

    def get_metric(self):
        output = dict()
        for key, val in self.out.items():
            metric = self.metrics[key].gather_measure(val)
            if key == self.key:
                output.update(metric)
            else:
                for sub_key in metric:
                    output['{}_{}'.format(key, sub_key)] = metric[sub_key]
        self.out.clear()
        return output

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()
