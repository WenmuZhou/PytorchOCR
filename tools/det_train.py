# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing

import os
import sys
import pathlib

# 将 torchocr路径加到python路径里
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

from torchocr.networks import build_model, build_loss
from torchocr.postprocess import build_post_process
from torchocr.datasets import build_dataloader
from torchocr.utils import get_logger, weight_init, load_checkpoint, save_checkpoint
from torchocr.metrics import DetMetric


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/det_train_db_config_local.py', help='train config file path')
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


def adjust_learning_rate(optimizer, base_lr, iter, all_iters, factor, warmup_iters=0, warmup_factor=1.0 / 3):
    """
    带 warmup 的学习率衰减
    :param optimizer: 优化器
    :param base_lr: 开始的学习率
    :param iter: 当前迭代次数
    :param all_iters: 总的迭代次数
    :param factor: 学习率衰减系数
    :param warmup_iters: warmup 迭代数
    :param warmup_factor: warmup 系数
    :return:
    """
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iter < warmup_iters:
        alpha = float(iter) / warmup_iters
        rate = warmup_factor * (1 - alpha) + alpha
    else:
        rate = np.power(1.0 - iter / float(all_iters + 1), factor)
    lr = rate * base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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


def evaluate(net, val_loader, to_use_device, logger, post_process, metric):
    """
    在验证集上评估模型

    :param net: 网络
    :param val_loader: 验证集 dataloader
    :param to_use_device: device
    :param logger: logger类对象
    :param post_process: 后处理类对象
    :param metric: 根据网络输出和 label 计算 acc 等指标的类对象
    :return:  一个包含 eval_loss，eval_acc和 norm_edit_dis 的 dict,
        例子： {
                'recall':0,
                'precision': 0.99,
                'hmean': 0.9999,
                }
    """
    logger.info('start evaluate')
    net.eval()
    raw_metrics = []
    total_frame = 0.0
    total_time = 0.0
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            start = time.time()
            output = net.forward(batch_data['img'].to(to_use_device))
            boxes, scores = post_process(output.cpu().numpy(), batch_data['shape'], is_output_polygon=metric.is_output_polygon)
            total_frame += batch_data['img'].size()[0]
            total_time += time.time() - start
            raw_metric = metric(batch_data, (boxes, scores))
            raw_metrics.append(raw_metric)
    metrics = metric.gather_measure(raw_metrics)
    net.train()
    result_dict = {'recall': metrics['recall'].avg, 'precision': metrics['precision'].avg,
                   'hmean': metrics['fmeasure'].avg}
    for k, v in result_dict.items():
        logger.info(f'{k}:{v}')
    logger.info('FPS:{}'.format(total_frame / total_time))
    return result_dict


def train(net, optimizer, loss_func, train_loader, eval_loader, to_use_device,
          cfg, global_state, logger, post_process):
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
    :param post_process: 后处理类对象
    :return: None
    """

    train_options = cfg.train_options
    metric = DetMetric()
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
        best_model = {'recall': 0, 'precision': 0, 'hmean': 0, 'best_model_epoch': 0}
        start_epoch = 0
        global_step = 0
    # 开始训练
    base_lr = cfg['optimizer']['lr']
    all_iters = all_step * train_options['epochs']
    warmup_iters = 3 * all_step
    try:
        for epoch in range(start_epoch, train_options['epochs']):  # traverse each epoch
            net.train()  # train mode
            train_loss = 0.
            start = time.time()
            for i, batch_data in enumerate(train_loader):  # traverse each batch in the epoch
                current_lr = adjust_learning_rate(optimizer, base_lr, global_step, all_iters, 0.9,
                                                  warmup_iters=warmup_iters)
                # 数据进行转换和丢到gpu
                for key, value in batch_data.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch_data[key] = value.to(to_use_device)
                # 清零梯度及反向传播
                optimizer.zero_grad()
                output = net.forward(batch_data['img'].to(to_use_device))
                loss_dict = loss_func(output, batch_data)
                loss_dict['loss'].backward()
                optimizer.step()
                # statistic loss for print
                train_loss += loss_dict['loss'].item()
                loss_str = 'loss: {:.4f} - '.format(loss_dict.pop('loss').item())
                for idx, (key, value) in enumerate(loss_dict.items()):
                    loss_dict[key] = value.item()
                    loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                    if idx < len(loss_dict) - 1:
                        loss_str += ' - '
                if (i + 1) % train_options['print_interval'] == 0:
                    interval_batch_time = time.time() - start
                    logger.info(f"[{epoch}/{train_options['epochs']}] - "
                                f"[{i + 1}/{all_step}] - "
                                f"lr:{current_lr} - "
                                f"{loss_str} - "
                                f"time:{interval_batch_time:.4f}")
                    start = time.time()
                global_step += 1
            logger.info(f'train_loss: {train_loss / len(train_loader)}')
            if (epoch + 1) % train_options['val_interval'] == 0:
                global_state['start_epoch'] = epoch
                global_state['best_model'] = best_model
                global_state['global_step'] = global_step
                net_save_path = f"{train_options['checkpoint_save_dir']}/latest.pth"
                save_checkpoint(net_save_path, net, optimizer, logger, cfg, global_state=global_state)
                if train_options['ckpt_save_type'] == 'HighestAcc':
                    # val
                    eval_dict = evaluate(net, eval_loader, to_use_device, logger, post_process, metric)
                    if eval_dict['hmean'] > best_model['hmean']:
                        best_model.update(eval_dict)
                        best_model['best_model_epoch'] = epoch
                        best_model['models'] = net_save_path

                        global_state['start_epoch'] = epoch
                        global_state['best_model'] = best_model
                        global_state['global_step'] = global_step
                        net_save_path = f"{train_options['checkpoint_save_dir']}/best.pth"
                        save_checkpoint(net_save_path, net, optimizer, logger, cfg, global_state=global_state)
                elif train_options['ckpt_save_type'] == 'FixedEpochStep' and epoch % train_options[
                    'ckpt_save_epoch'] == 0:
                    shutil.copy(net_save_path, net_save_path.replace('latest.pth', f'{epoch}.pth'))
                best_str = 'current best, '
                for k, v in best_model.items():
                    best_str += '{}: {}, '.format(k, v)
                logger.info(best_str)
    except KeyboardInterrupt:
        import os
        save_checkpoint(os.path.join(train_options['checkpoint_save_dir'], 'final.pth'), net, optimizer, logger, cfg,
                        global_state=global_state)
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

    # ===> whether to resume from checkpoint
    resume_from = train_options['resume_from']
    if resume_from:
        net, _resumed_optimizer, global_state = load_checkpoint(net, resume_from, to_use_device, optimizer,
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
    train_loader = build_dataloader(cfg.dataset.train)
    eval_loader = build_dataloader(cfg.dataset.eval)

    # post_process
    post_process = build_post_process(cfg['post_process'])
    # ===> train
    train(net, optimizer, loss_func, train_loader, eval_loader, to_use_device, cfg, global_state, logger, post_process)


if __name__ == '__main__':
    main()
