import copy

import torch

__all__ = ['build_optimizer']


def build_optimizer(optim_config, lr_scheduler_config, epochs, step_each_epoch, model):
    from . import lr
    config = copy.deepcopy(optim_config)
    optim = getattr(torch.optim, config.pop('name'))(params=model.parameters(), **config)

    lr_config = copy.deepcopy(lr_scheduler_config)
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    lr_scheduler = getattr(lr, lr_config.pop('name'))(**lr_config)(optimizer=optim)
    return optim, lr_scheduler
