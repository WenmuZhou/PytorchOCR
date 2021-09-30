import os
import copy
import torch
from torch import nn
from addict import Dict

from .DetModel import DetModel
from addict import Dict as AttrDict

__all__ = ['DistillationModel']


def load_pretrained_params(_model, _path):
    if _path is None:
        return False
    if not os.path.exists(_path):
        print(f'The pretrained_model {_path} does not exists')
        return False
    params = torch.load(_path)
    state_dict = params['state_dict']
    state_dict_no_module = {k.replace('module.', ''): v for k, v in state_dict.items()}
    _model.load_state_dict(state_dict_no_module)
    return _model


class DistillationModel(nn.Module):
    def __init__(self, config):
        super(DistillationModel, self).__init__()
        self.model_dict = nn.ModuleDict()
        self.model_name_list = []

        sub_model_cfgs = config['models']
        for key in sub_model_cfgs:
            sub_cfg = copy.deepcopy(sub_model_cfgs[key])
            sub_cfg.pop('type')
            freeze_params = False
            pretrained = None

            if 'freeze_params' in sub_cfg:
                freeze_params = sub_cfg.pop('freeze_params')
            if 'pretrained' in sub_cfg:
                pretrained = sub_cfg.pop('pretrained')
            model = DetModel(Dict(sub_cfg))
            if pretrained is not None:
                model = load_pretrained_params(model, pretrained)
            if freeze_params:
                for para in model.parameters():
                    para.requires_grad = False
                model.training = False

            self.model_dict[key] = model
            self.model_name_list.append(key)

    def forward(self, x):
        result_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            result_dict[model_name] = self.model_dict[model_name](x)
        return result_dict
