# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing

from tools.train_config import *
import torch
import numpy as np
import random
import importlib


def set_random_seed(seed, use_cuda=True, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        use_cuda: whether depend on cuda
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_architecture(arch_name, **kwargs):
    """
    get architecture model class
    """
    assert arch_name in {'CRNNMBV3', 'CRNNRes34'}, f'{arch_name} is not developed yet!'
    module = importlib.import_module(f'torchocr.networks.architectures.RecModels')
    arch_model = getattr(module, arch_name)
    return arch_model(**kwargs)


def main():
    logger.info(f'=>train options:\n\t{train_options}')
    to_use_device = torch.device(
        device if torch.cuda.is_available() and use_cuda else 'cpu')
    set_random_seed(SEED, use_cuda, deterministic=True)

    # ===> build network
    net = get_architecture(architecture, **architecture_config).to(to_use_device)

    # ===> whether to resume from checkpoint
    if resume_from:
        if to_use_device.type == 'cpu':
            net.load_state_dict(torch.load(resume_from, map_location=to_use_device))
        else:
            net.load_state_dict(torch.load(resume_from))
        logger.info(f'==> net resume from {resume_from}')
    else:
        logger.info(f'==> net resume from scratch.')

    # ===> loss function

    # ===> solver

    # ===> data loader
    dataset = build_dataset()

    # ===> train
    train_model(model, dataset, )


if __name__ == '__main__':
    main()
