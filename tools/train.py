# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing


import os
import sys
from importlib import import_module
import random
from torch import nn
import numpy as np
import torch
import torch.optim as optim
from torch.optim import SGD
from addict import Dict
from utils import weight_init, StrLabelConverter, init_logger

import argparse


def parse_args(logger):
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('config', type=str, help='train config file path')
    args = parser.parse_args()
    # 解析.py文件
    config_path = os.path.abspath(os.path.expanduser(args.config))
    if not os.path.isfile(config_path):
        logger.error(f'{config_path} does not exist!')
    if config_path.endswith('.py'):
        module_name = os.path.basename(config_path)[:-3]
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        return cfg_dict
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


def get_architecture(arch_config):
    """
    get architecture model class
    """
    arch_type = arch_config.pop('type')
    assert arch_type in {'RecModel', 'DetModel'}, f'{arch_type} is not developed yet!'
    module = import_module(f'torchocr.networks.architectures.{arch_type}')
    arch_model = getattr(module, arch_type)
    if not arch_config:
        return arch_model()
    else:
        return arch_model(Dict(arch_config))


def get_loss(loss_config):
    """
    get loss function
    Args:
        loss_config:
    Returns:
    """
    loss_type = loss_config.pop('type')
    # assert loss_type in {'CTCLoss'}, f'{loss_type} is not developed yet!'
    module = import_module(f'torchocr.networks.losses.{loss_type}')
    arch_model = getattr(module, loss_type)
    if not loss_config:
        return arch_model()
    else:
        return arch_model(**loss_config)


def load_model(_model, resume_from, to_use_device, optimizer=None, third_name=None):
    """
    加载预训练模型
    Args:
        _model:  模型
        resume_from: 预训练模型路径
        to_use_device: 设备
        optimizer: 如果不为None，则表明采用模型的训练参数
        third_name:

    Returns:

    """
    start_epoch = 0
    if not third_name:
        state = torch.load(resume_from, map_location=to_use_device)
        _model.load_state_dict(state)
        if optimizer is not None:
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch']

    elif third_name == 'paddle':
        pass
    return _model, start_epoch, optimizer


def get_optimizers(params, rec_train_options):
    """
    优化器
    Returns:
    """
    return [SGD(params, lr=rec_train_options['base_lr'], weight_decay=rec_train_options['weight_decay'])]


def get_lrs(optimizer, type='LambdaLR', **kwargs):
    """
    """
    scheduler = None
    if type == 'LambdaLR':
        burn_in, steps = kwargs['burn_in'], kwargs['steps']

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
    elif type == 'StepLR':
        # 等间隔调整学习率， 调整倍数为gamma倍，调整间隔为step_size，间隔单位是step，step通常是指epoch。
        step_size, gamma = kwargs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif type == 'ReduceLROnPlateau':
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
    for stage in finetune_stage:
        attr = getattr(net, stage, None)
        for element in attr.parameters():
            yield element


def get_data_loader(dataset_config, batch_size):
    """
    数据加载
    :return:
    """
    dataset_type = dataset_config.pop('type')
    train_dataset_cfg =  dataset_config.pop('train')
    eval_dataset_cfg = dataset_config.pop('eval')
    assert dataset_type in {'ICDAR15RecDataset', 'ICDAR15DetDataset'}, f'{dataset_type} is not developed yet!'
    train_module = import_module(f'dataset.icdar2015.{dataset_type}')
    eval_module = import_module(f'dataset.icdar2015.{dataset_type}')
    train_dataset_class = getattr(train_module, dataset_type)
    eval_dataset_class = getattr(eval_module, dataset_type)
    # 此处需要转换，讨论转换为什么格式
    train_set = train_dataset_class(Dict(train_dataset_cfg))
    eval_set = eval_dataset_class(Dict(eval_dataset_cfg))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1,  # one image each batch for testing
                                              shuffle=False, num_workers=4,
                                              pin_memory=True)
    return train_loader, eval_loader


def evaluate(net, val_loader, loss_func, to_use_device, logger, max_iter=50):

    logger.info('start val')
    net.eval()
    total_loss = 0.0
    k = 0
    correct_num = 0
    total_num = 0
    val_iter = iter(val_loader)
    max_iter = min(max_iter, len(val_loader))
    for i in range(max_iter):
        k = k + 1
        (data, label) = val_iter.next()
        data, label = data.to(to_use_device), label.to(to_use_device)
        output = net.forward(data)
        loss = loss_func(output, label)
        total_loss += float(loss)
        pred_label = output.max(2)[1]
        pred_label = pred_label.transpose(1, 0).contiguous().view(-1)
        total_num += len(pred_label)
        for x, y in zip(pred_label, label):
            if int(x) == int(y):
                correct_num += 1
    accuracy = correct_num / float(total_num) * 100
    evaluate_loss = total_loss / k
    logger.info('evaluate loss : %.3f , accuary : %.3f%%' % (evaluate_loss, accuracy))


def save_model(checkpoint_path, model, solver, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': solver.state_dict(),
             'epoch': epoch}
    torch.save(state, checkpoint_path)
    logger.info('models saved to %s' % checkpoint_path)


def save_model_logic(total_loss, total_num, min_loss, net, solver, epoch, rec_train_options, logger):
    """
    根据配置文件保存模型
    Args:
        total_loss:
        total_num:
        min_loss:
        net:
        epoch:
        rec_train_options:
        logger:
    Returns:

    """
    # operation for model save as parameter ckpt_save_type is  HighestAcc
    if rec_train_options['ckpt_save_type'] == 'HighestAcc':
        loss_mean = sum([total_loss[idx] * total_num[idx] for idx in range(len(total_loss))]) / sum(total_num)
        if loss_mean < min_loss:
            min_loss = loss_mean
            save_model(os.path.join(rec_train_options['checkpoint_save_dir'], 'epoch_' + str(epoch) + '.pth'), net,
                       solver, epoch, logger)

    else:
        if epoch % rec_train_options['ckpt_save_epoch'] == 0:
            save_model(os.path.join(rec_train_options['checkpoint_save_dir'], 'epoch_' + str(epoch) + '.pth'), net,
                       solver, epoch, logger)
    return min_loss


def train(net, solver, scheduler, loss_func, train_loader, eval_loader, to_use_device,
          rec_train_options, logger):
    """
    训练
    Args:
        net: 模型
        solver: 优化器
        scheduler: 学习率更新
        loss_func: loss函数
        train_loader: 训练数据集dataloader
        eval_loader: 验证数据集dataloader
        to_use_device:
        rec_train_options:
        logger:
    Returns:
    """

    # ===>
    use_cuda = torch.cuda.is_available() and ('cuda' in rec_train_options['device'])

    logger.info('==> Training...')
    net.train()  # train mode

    # ===> print loss信息的参数
    loss_for_print = 0.0
    num_in_print = 0
    current_step = 0

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_loader.__len__(), all_step))
    # ===> parameter for model save
    min_loss = 10000
    epoch = 0
    try:
        for epoch in range(rec_train_options['epochs']):  # traverse each epoch
            total_loss = []
            total_num = []
            for i, (data, labels, ) in enumerate(train_loader):  # traverse each batch in the epoch
                current_step = epoch * all_step + i
                num_in_print = num_in_print + 1
                # put training data, label to device
                data, labels = data.to(to_use_device), labels.to(to_use_device)

                # forward calculation
                output = net.forward(data)

                # calculate  loss
                loss = loss_func(output, labels)

                if i % rec_train_options['print_interval'] == 0:
                    # display
                    logger.info("[%d/%d] || [%d/%d] || Loss:%.3f" % (
                        epoch, rec_train_options['epochs'], i + 1, all_step, loss_for_print / num_in_print))
                    # operation for model save as parameter ckpt_save_type is  HighestAcc
                    if rec_train_options['ckpt_save_type'] == 'HighestAcc':
                        total_loss.append(loss_for_print)
                        total_num.append(num_in_print)
                    loss_for_print = 0.0
                    num_in_print = 0

                # clear the grad
                solver.zero_grad()

                loss.backward()
                solver.step()
                scheduler.step()
                if i % rec_train_options['val_interval'] == 0:
                    # val
                    evaluate(net, eval_loader, loss_func, to_use_device, logger)
                    net.train()  # train mode
            # 保存ckpt
            # operation for model save as parameter ckpt_save_type is  HighestAcc
            min_loss = save_model_logic(total_loss, total_num, min_loss, net, epoch, rec_train_options, logger)
    except KeyboardInterrupt:
        save_model(os.path.join(rec_train_options['checkpoint_save_dir'], 'final_' + str(epoch) + '.pth'), net, solver,
                   epoch, logger)
    finally:
        pass


def train_info_initial(rec_train_options, logger):
    """

    Args:
        rec_train_options:
        logger:
    Returns:
    """
    logger.info('=>train options:')
    for key, val in rec_train_options.items():
        logger.info(f'\t{key} : {val}')
    if rec_train_options['checkpoint_save_dir']:
        os.makedirs(rec_train_options['checkpoint_save_dir'], exist_ok=True)
    else:
        os.makedirs('./checkpoint_dir', exist_ok=True)


def main():
    # ===> 获取配置文件参数
    logger = init_logger()
    cfg = parse_args(logger)

    # ===> 训练信息的打印
    rec_train_options = cfg['rec_train_options']
    train_info_initial(rec_train_options, logger)

    # ===>
    to_use_device = torch.device(
        rec_train_options['device'] if torch.cuda.is_available() and ('cuda' in rec_train_options['device']) else 'cpu')
    set_random_seed(cfg['SEED'], 'cuda' in rec_train_options['device'], deterministic=True)

    # ===> build network
    net = get_architecture(cfg['model'])

    # ===> 模型初始化及模型部署到对应的设备
    net.apply(weight_init)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.to(to_use_device)

    # ===> get fine tune layers
    params_to_train = get_fine_tune_params(net, rec_train_options['fine_tune_stage'])
    # ===> solver and lr scheduler
    solver = get_optimizers(params_to_train, rec_train_options)
    scheduler = get_lrs(solver[0], rec_train_options['lr_scheduler'], **rec_train_options['lr_scheduler_info'])

    # ===> whether to resume from checkpoint
    resume_from = rec_train_options['resume_from']
    if resume_from:
        net, _, solver = load_model(net, resume_from, to_use_device)
        logger.info(f'==> net resume from {resume_from}')
    else:
        logger.info(f'==> net resume from scratch.')

    # ===> loss function
    loss_func = get_loss(cfg['loss'])
    if torch.cuda.is_available and ('cuda' in rec_train_options['device']):
        loss_func = loss_func.cuda()

    # ===> data loader
    train_loader, eval_loader = get_data_loader(cfg['dataset'], rec_train_options['batch_size'])

    # ===> train
    train(net, solver[0], scheduler, loss_func, train_loader, eval_loader, to_use_device, rec_train_options, logger)


if __name__ == '__main__':
    main()
