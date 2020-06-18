# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing

import os
import sys
import pathlib

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(str(__dir__.parent.parent))
import random
from importlib import import_module

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from torchocr.networks import build_model, build_loss
from torchocr.datasets import build_dataloader
from torchocr.utils import get_logger, weight_init, load_checkpoint


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/rec_train_lmdb_config.py', help='train config file path')
    args = parser.parse_args()
    # 解析.py文件
    config_path = os.path.abspath(os.path.expanduser(args.config))
    assert os.path.isfile(config_path)
    if config_path.endswith('.py'):
        module_name = os.path.basename(config_path)[:-3]
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        return mod.config
        # cfg_dict = {
        #     name: value
        #     for name, value in mod.__dict__.items()
        #     if not name.startswith('__')
        # }
        # return cfg_dict
    else:
        raise IOError('Only py type are supported now!')


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


def build_optimizer(params, config):
    """
    优化器
    Returns:
    """
    from torch import optim
    opt_type = config.pop('type')
    opt = getattr(optim, opt_type)(params, **config)
    return opt


def build_scheduler(optimizer, config):
    """
    """
    scheduler = None
    sch_type = config.pop('type')
    if sch_type == 'LambdaLR':
        burn_in, steps = config['burn_in'], config['steps']

        # Learning rate setup
        def burnin_schedule(i):
            if i < burn_in:
                factor = pow(i / burn_in, 4)
            elif i < steps[0]:
                factor = 1.0
            elif i < steps[1]:
                factor = 0.1
            else:
                factor = 0.01
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    elif sch_type == 'StepLR':
        # 等间隔调整学习率， 调整倍数为gamma倍，调整间隔为step_size，间隔单位是step，step通常是指epoch。
        step_size, gamma = config['step_size'], config['gamma']
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sch_type == 'ReduceLROnPlateau':
        # 当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。例如，当验证集的loss不再下降时，进行学习率调整；或者监测验证集的accuracy，当accuracy不再上升时，则调整学习率。
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=3, verbose=True, threshold=1e-4)
    return scheduler


def get_fine_tune_params(net, finetune_stage):
    """
    获取需要优化的参数
    Args:
        net:
    Returns: 需要优化的参数
    """
    to_return_parameters = []
    for stage in finetune_stage:
        attr = getattr(net.module, stage, None)
        for element in attr.parameters():
            to_return_parameters.append(element)
    return to_return_parameters


def main():
    # ===> 获取配置文件参数
    cfg = parse_args()
    os.makedirs(cfg.train_options['checkpoint_save_dir'], exist_ok=True)
    logger = get_logger('torchocr', log_file=os.path.join(cfg.train_options['checkpoint_save_dir'], 'train.log'))

    # ===> 训练信息的打印
    train_options = cfg.train_options
    logger.info(cfg)
    # ===>
    to_use_device = torch.device(
        train_options['device'] if torch.cuda.is_available() and ('cuda' in train_options['device']) else 'cpu')
    set_random_seed(cfg['SEED'], 'cuda' in train_options['device'], deterministic=True)

    # ===> build network
    net = build_model(cfg['model'])

    # ===> 模型初始化及模型部署到对应的设备
    net.apply(weight_init)
    # if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
    net = net.to(to_use_device)
    net.train()

    # ===> get fine tune layers
    params_to_train = get_fine_tune_params(net, train_options['fine_tune_stage'])
    # ===> solver and lr scheduler
    optimizer = build_optimizer(net.parameters(), cfg['optimizer'])
    scheduler = build_scheduler(optimizer, cfg['lr_scheduler'])

    # ===> whether to resume from checkpoint
    resume_from = train_options['resume_from']
    if resume_from:
        net, current_epoch, _resumed_optimizer = load_checkpoint(net, resume_from, to_use_device, optimizer,
                                                                 third_name=train_options['third_party_name'])
        if _resumed_optimizer:
            optimizer = _resumed_optimizer
        logger.info(f'net resume from {resume_from}')
    else:
        current_epoch = 0
        logger.info(f'net resume from scratch.')

    # ===> loss function
    loss_func = build_loss(cfg['loss'])
    loss_func = loss_func.to(to_use_device)

    # ===> data loader
    train_loader = build_dataloader(cfg.dataset.train)
    eval_loader = build_dataloader(cfg.dataset.eval)

    # ===> train
    if net.module.name.startswith('Rec'):
        from tools.rec_train import train
    else:
        pass
    train(net, optimizer, scheduler, loss_func, train_loader, eval_loader, to_use_device, cfg, current_epoch, logger)


if __name__ == '__main__':
    main()
