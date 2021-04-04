# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing

import os
import sys
import pathlib

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
import random
import time
import shutil
import traceback
from importlib import import_module

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torchocr.networks import build_model, build_loss
from torchocr.datasets import build_dataloader
from torchocr.utils import get_logger, weight_init, load_checkpoint, save_checkpoint


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/rec_train_config.py', help='train config file path')
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


def evaluate(net, val_loader, loss_func, to_use_device, logger, converter, metric):
    """
    在验证集上评估模型

    :param net: 网络
    :param val_loader: 验证集 dataloader
    :param loss_func: 损失函数
    :param to_use_device: device
    :param logger: logger类对象
    :param converter: label转换器类对象
    :param metric: 根据网络输出和 label 计算 acc 等指标的类对象
    :return:  一个包含 eval_loss，eval_acc和 norm_edit_dis 的 dict,
        例子： {
                'eval_loss':0,
                'eval_acc': 0.99,
                'norm_edit_dis': 0.9999,
                }
    """
    logger.info('start evaluate')
    net.eval()
    nums = 0
    result_dict = {'eval_loss': 0., 'eval_acc': 0., 'norm_edit_dis': 0.}
    show_str = []
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            targets, targets_lengths = converter.encode(batch_data['label'])
            batch_data['targets'] = targets
            batch_data['targets_lengths'] = targets_lengths
            output = net.forward(batch_data['img'].to(to_use_device))
            loss = loss_func(output, batch_data)

            nums += batch_data['img'].shape[0]
            acc_dict = metric(output, batch_data['label'])
            result_dict['eval_loss'] += loss['loss'].item()
            result_dict['eval_acc'] += acc_dict['n_correct']
            result_dict['norm_edit_dis'] += acc_dict['norm_edit_dis']
            show_str.extend(acc_dict['show_str'])

    result_dict['eval_loss'] /= len(val_loader)
    result_dict['eval_acc'] /= nums
    result_dict['norm_edit_dis'] = 1 - result_dict['norm_edit_dis'] / nums
    logger.info(f"eval_loss:{result_dict['eval_loss']}")
    logger.info(f"eval_acc:{result_dict['eval_acc']}")
    logger.info(f"norm_edit_dis:{result_dict['norm_edit_dis']}")

    for s in show_str[:10]:
        logger.info(s)
    net.train()
    return result_dict


def train(net, optimizer, scheduler, loss_func, train_loader, eval_loader, to_use_device,
          cfg, global_state, logger):
    """
    训练函数

    :param net: 网络
    :param optimizer: 优化器
    :param scheduler: 学习率更新器
    :param loss_func: loss函数
    :param train_loader: 训练数据集 dataloader
    :param eval_loader: 验证数据集 dataloader
    :param to_use_device: device
    :param cfg: 当前训练所使用的配置
    :param global_state: 训练过程中的一些全局状态，如cur_epoch,cur_iter,最优模型的相关信息
    :param logger: logger 对象
    :return: None
    """

    from torchocr.metrics import RecMetric
    from torchocr.utils import CTCLabelConverter
    converter = CTCLabelConverter(cfg.dataset.alphabet)
    train_options = cfg.train_options
    metric = RecMetric(converter)
    # ===>
    logger.info('Training...')
    # ===> print loss信息的参数
    all_step = len(train_loader)
    logger.info(f'train dataset has {train_loader.dataset.__len__()} samples,{all_step} in dataloader')
    logger.info(f'eval dataset has {eval_loader.dataset.__len__()} samples,{len(eval_loader)} in dataloader')
    if len(global_state) > 0:
        best_model = global_state['best_model']
        start_epoch = global_state['start_epoch']
        global_step = global_state['global_step']
    else:
        best_model = {'best_acc': 0, 'eval_loss': 0, 'model_path': '', 'eval_acc': 0., 'eval_ned': 0.}
        start_epoch = 0
        global_step = 0
    # 开始训练
    try:
        for epoch in range(start_epoch, train_options['epochs']):  # traverse each epoch
            net.train()  # train mode
            start = time.time()
            for i, batch_data in enumerate(train_loader):  # traverse each batch in the epoch
                current_lr = optimizer.param_groups[0]['lr']
                cur_batch_size = batch_data['img'].shape[0]
                targets, targets_lengths = converter.encode(batch_data['label'])
                batch_data['targets'] = targets
                batch_data['targets_lengths'] = targets_lengths
                # 清零梯度及反向传播
                optimizer.zero_grad()
                output = net.forward(batch_data['img'].to(to_use_device))
                loss_dict = loss_func(output, batch_data)
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
                # statistic loss for print
                acc_dict = metric(output, batch_data['label'])
                acc = acc_dict['n_correct'] / cur_batch_size
                norm_edit_dis = 1 - acc_dict['norm_edit_dis'] / cur_batch_size
                if (i + 1) % train_options['print_interval'] == 0:
                    interval_batch_time = time.time() - start
                    logger.info(f"[{epoch}/{train_options['epochs']}] - "
                                f"[{i + 1}/{all_step}] - "
                                f"lr:{current_lr} - "
                                f"loss:{loss_dict['loss'].item():.4f} - "
                                f"acc:{acc:.4f} - "
                                f"norm_edit_dis:{norm_edit_dis:.4f} - "
                                f"time:{interval_batch_time:.4f}")
                    start = time.time()
                if (i + 1) >= train_options['val_interval'] and (i + 1) % train_options['val_interval'] == 0:
                    global_state['start_epoch'] = epoch
                    global_state['best_model'] = best_model
                    global_state['global_step'] = global_step
                    net_save_path = f"{train_options['checkpoint_save_dir']}/latest.pth"
                    save_checkpoint(net_save_path, net, optimizer, logger, cfg, global_state=global_state)
                    if train_options['ckpt_save_type'] == 'HighestAcc':
                        # val
                        eval_dict = evaluate(net, eval_loader, loss_func, to_use_device, logger, converter, metric)
                        if eval_dict['eval_acc'] > best_model['eval_acc']:
                            best_model.update(eval_dict)
                            best_model['best_model_epoch'] = epoch
                            best_model['models'] = net_save_path

                            global_state['start_epoch'] = epoch
                            global_state['best_model'] = best_model
                            global_state['global_step'] = global_step
                            net_save_path = f"{train_options['checkpoint_save_dir']}/best.pth"
                            save_checkpoint(net_save_path, net, optimizer, logger, cfg, global_state=global_state)
                    elif train_options['ckpt_save_type'] == 'FixedEpochStep' and epoch % train_options['ckpt_save_epoch'] == 0:
                        shutil.copy(net_save_path, net_save_path.replace('latest.pth', f'{epoch}.pth'))
                global_step += 1
            scheduler.step()
    except KeyboardInterrupt:
        import os
        save_checkpoint(os.path.join(train_options['checkpoint_save_dir'], 'final.pth'), net, optimizer, logger, cfg, global_state=global_state)
    except:
        error_msg = traceback.format_exc()
        logger.error(error_msg)
    finally:
        for k, v in best_model.items():
            logger.info(f'{k}: {v}')


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
    if not cfg['model']['backbone']['pretrained']:  # 使用 pretrained
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
        net, _resumed_optimizer,global_state = load_checkpoint(net, resume_from, to_use_device, optimizer,
                                                                 third_name=train_options['third_party_name'])
        if _resumed_optimizer:
            optimizer = _resumed_optimizer
        logger.info(f'net resume from {resume_from}')
    else:
        global_state = {}
        logger.info(f'net resume from scratch.')

    # ===> loss function
    loss_func = build_loss(cfg['loss'])
    loss_func = loss_func.to(to_use_device)


    # ===> data loader
    cfg.dataset.train.dataset.alphabet = cfg.dataset.alphabet
    train_loader = build_dataloader(cfg.dataset.train)
    cfg.dataset.eval.dataset.alphabet = cfg.dataset.alphabet
    eval_loader = build_dataloader(cfg.dataset.eval)

    # ===> train
    train(net, optimizer, scheduler, loss_func, train_loader, eval_loader, to_use_device, cfg, global_state, logger)


if __name__ == '__main__':
    main()
