import torch
import torch.nn as nn
from .distillation_loss import DistillationDilaDBLoss,DistillationDBLoss,DistillationDMLLoss

class CombinedLoss(nn.Module):
    def __init__(self, _cfg_list=None):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        for key, val in _cfg_list['combine_list'].items():
            self.loss_weight.append(val.pop('weight'))
            self.loss_func.append(eval(key)(**val))

    def forward(self, input, batch, **kwargs):
        loss_dict = {}
        loss_all = 0.
        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch, **kwargs)
            weight = self.loss_weight[idx]
            loss = {key: loss[key] * weight for key in loss}
            if 'loss' in loss:
                loss_all =torch.add(loss_all, loss['loss'])
            else:
                loss_all += torch.add(list(loss.values()))
            loss_dict.update(loss)
        loss_dict['loss'] = loss_all
        return loss_dict
