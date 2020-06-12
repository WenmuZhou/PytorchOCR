# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing


import argparse
import os
import random
import sys
from importlib import import_module

import numpy as np
import torch
import torch.optim as optim
import tqdm
from addict import Dict
from torch import nn
from torch.optim import SGD, Adam
import traceback as tb

from datasets.icdar15.ICDAR15RecDataset import RecDataLoader
from utils import weight_init, init_logger


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


def load_model(_model, resume_from, to_use_device, _optimizers=None, third_name=None):
    """
    加载预训练模型
    Args:
        _model:  模型
        resume_from: 预训练模型路径
        to_use_device: 设备
        _optimizers: 如果不为None，则表明采用模型的训练参数
        third_name: 第三方预训练模型的名称

    Returns:

    """
    start_epoch = 0
    if not third_name:
        state = torch.load(resume_from, map_location=to_use_device)
        _model.load_state_dict(state['state_dict'])
        if _optimizers is not None:
            assert len(_optimizers) == len(state['optimizer'])
            for m_optimizer,m_optimizer_state_dict in zip(_optimizers,state['optimizer']):
                m_optimizer.load_state_dict(m_optimizer_state_dict)
        start_epoch = state['epoch']

    elif third_name == 'paddle':
        import paddle.fluid as fluid
        paddle_model = fluid.io.load_program_state(resume_from)
        _model.load_3rd_state_dict(third_name, paddle_model)
    return _model, start_epoch, _optimizers


def get_optimizers(params, rec_train_options):
    """
    优化器
    Returns:
    """
    return [Adam(params, lr=rec_train_options['base_lr'], weight_decay=rec_train_options['weight_decay'])]


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
        step_size, gamma = kwargs['step_size'], kwargs['gamma']
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
    to_return_parameters = []
    for stage in finetune_stage:
        attr = getattr(net, stage, None)
        for element in attr.parameters():
            to_return_parameters.append(element)
    return to_return_parameters


def get_data_loader(dataset_config):
    """
    数据加载
    :return:
    """
    dataset_type = dataset_config.pop('type')
    train_dataset_cfg = dataset_config.pop('train')
    eval_dataset_cfg = dataset_config.pop('eval')
    assert dataset_type in {'icdar15.ICDAR15RecDataset',
                            'icdar15.ICDAR15DetDataset'}, \
        f'{dataset_type} is not developed yet!'
    train_module = import_module(f'datasets.{dataset_type}')
    eval_module = import_module(f'datasets.{dataset_type}')
    train_dataset_class = getattr(train_module, dataset_type.split('.')[-1])
    eval_dataset_class = getattr(eval_module, dataset_type.split('.')[-1])
    # 此处需要转换，讨论转换为什么格式
    train_set = train_dataset_class(Dict(train_dataset_cfg))
    eval_set = eval_dataset_class(Dict(eval_dataset_cfg))
    train_loader = RecDataLoader(train_set, Dict(train_dataset_cfg))
    eval_loader = RecDataLoader(eval_set, Dict(eval_dataset_cfg))
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=4, pin_memory=True)
    # eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1,  # one image each batch for testing
    #                                           shuffle=False, num_workers=4,
    #                                           pin_memory=True)
    return train_loader, eval_loader


def evaluate(net, val_loader, loss_func, to_use_device, logger, max_iter=50):
    logger.info('start evaluate')
    net.eval()
    nums = 0
    result_dict = {
        'eval_loss': 0.,
        'recall': 0.,
        'precision': 0.,
        'f1': 0.
    }
    with torch.no_grad():
        for batch_data in tqdm.tqdm(val_loader):
            # for data in batch_data:
            #     data.to(to_use_device)
            output = net.forward(batch_data[0].to(to_use_device))
            loss = loss_func(output, batch_data[1:])
            # print('eval loss {}'.format(float(loss.item())))
            result_dict['eval_loss'] += loss.item()*batch_data[0].size(0)
            # res = cal_recognize_recall_precision_f1(output, batch_data[1:])
            #
            # result_dict['recall'] += res['recall']
            # result_dict['precision'] += res['precision']
            # result_dict['f1'] += res['f1']
            # print('batch shape:{}'.format(batch_data[0].shape[0]))
            nums += batch_data[0].shape[0]
    logger.info(f'evaluate result:\n\t nums:{nums}')
    assert nums > 0, 'there is no eval data available'
    for key, val in result_dict.items():
        result_dict[key] = result_dict[key] / nums
        logger.info('\t {}:{}'.format(key, result_dict[key]))
    return result_dict['eval_loss'], result_dict['recall'], result_dict['precision'], result_dict['f1']


def save_checkpoint(checkpoint_path, model, _optimizers, epoch, logger):
    state = {'state_dict': model.state_dict(),
             'optimizer': [_.state_dict() for _ in _optimizers],
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
            save_checkpoint(os.path.join(rec_train_options['checkpoint_save_dir'], 'epoch_' + str(epoch) + '.pth'), net,
                            solver, epoch, logger)

    else:
        if epoch % rec_train_options['ckpt_save_epoch'] == 0:
            save_checkpoint(os.path.join(rec_train_options['checkpoint_save_dir'], 'epoch_' + str(epoch) + '.pth'), net,
                            solver, epoch, logger)
    return min_loss


def train(net, _solvers, schedulers, loss_func, train_loader, eval_loader, to_use_device,
          rec_train_options,_epoch, logger):
    """
    训练
    Args:
        net: 模型
        _solvers: 优化器
        schedulers: 学习率更新
        loss_func: loss函数
        train_loader: 训练数据集dataloader
        eval_loader: 验证数据集dataloader
        to_use_device:
        rec_train_options:
        _epoch: 当前epoch
        logger:
    Returns:
    """

    # ===>
    logger.info('==> Training...')
    # ===> print loss信息的参数
    loss_for_print = 0.0
    num_in_print = 0
    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_loader.__len__(), all_step))
    best_model = {'eval_loss': 0, 'recall': 0, 'precision': 0, 'f1': 0, 'models': ''}
    try:
        for epoch in range(_epoch,rec_train_options['epochs']):  # traverse each epoch
            current_lr = [m_scheduler.get_lr()[0] for m_scheduler in schedulers]
            net.train()  # train mode
            for i, batch_data in enumerate(train_loader):  # traverse each batch in the epoch
                # clear the grad
                [_solver.zero_grad() for _solver in _solvers]
                # forward calculation
                output = net.forward(batch_data[0].to(to_use_device))
                # calculate  loss
                loss = loss_func(output, batch_data[1:])
                # statistic loss for print
                loss_for_print = loss_for_print + loss.item()
                num_in_print = num_in_print + 1

                if i % rec_train_options['print_interval'] == 0:
                    # interval_batch_time = time.time() - start
                    # display
                    logger.info("[%d/%d] || [%d/%d] ||lr:%s || mean loss for batch:%.3f || " % (
                        epoch, rec_train_options['epochs'], i + 1, all_step,
                        ','.join(['%0.4f' % m_lr for m_lr in current_lr]),
                        loss_for_print / num_in_print))
                    loss_for_print = 0.0
                    num_in_print = 0
                loss.backward()
                [_solver.step() for _solver in _solvers]

            if epoch >= rec_train_options['val_interval'] and epoch % rec_train_options['val_interval'] == 0:
                # val
                eval_loss, recall, precision, f1 = evaluate(net, eval_loader, loss_func, to_use_device, logger)
                net_save_path = '{}/epoch_{}_eval_loss{:.6f}_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(
                    rec_train_options['checkpoint_save_dir'], epoch, eval_loss, recall, precision, f1)
                # save_checkpoint(net_save_path, net, solver, epoch, logger)

                if eval_loss > best_model['eval_loss']:
                    # best_path = glob.glob(rec_train_options['checkpoint_save_dir'] + '/Best_*.pth')
                    # for b_path in best_path:
                    #     if os.path.exists(b_path):
                    #         os.remove(b_path)
                    best_model['eval_loss'] = eval_loss
                    best_model['precision'] = precision
                    best_model['f1'] = f1
                    best_model['models'] = net_save_path
                    save_checkpoint(net_save_path, net, _solvers, epoch, logger)
                    # best_save_path = '{}/Best_{}_r{:.6f}_p{:.6f}_f1{:.6f}.pth'.format(
                    #     rec_train_options['checkpoint_save_dir'], epoch,
                    #     recall,
                    #     precision,
                    #     f1)
                    # if os.path.exists(net_save_path):
                    #     shutil.copyfile(net_save_path, best_save_path)
                    # else:
                    #     save_checkpoint(best_save_path, net, solver, epoch, logger)
                    #
                    # pse_path = glob.glob(rec_train_options['checkpoint_save_dir'] + '/PSENet_*.pth')
                    # for p_path in pse_path:
                    #     if os.path.exists(p_path):
                    #         os.remove(p_path)
            # # 保存ckpt
            # # operation for model save as parameter ckpt_save_type is  HighestAcc
            # min_loss = save_model_logic(total_loss, total_num, min_loss, net, epoch, rec_train_options, logger)
            [_scheduler.step() for _scheduler in schedulers]

    except KeyboardInterrupt:
        save_checkpoint(os.path.join(rec_train_options['checkpoint_save_dir'], 'final_' + str(epoch) + '.pth'), net,
                        _solvers, epoch, logger)
    except Exception as e:
        tb.print_exc(limit=1, file=sys.stdout)
        logger.error(str(e))
    finally:
        if best_model['models']:
            logger.info(best_model)


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
    net.train()

    # ===> get fine tune layers
    params_to_train = get_fine_tune_params(net, rec_train_options['fine_tune_stage'])
    # ===> solver and lr scheduler
    solvers = get_optimizers(params_to_train, rec_train_options)
    schedulers = [get_lrs(m_solver, rec_train_options['lr_scheduler'], **rec_train_options['lr_scheduler_info']) for
                  m_solver in solvers]

    # ===> whether to resume from checkpoint
    resume_from = rec_train_options['resume_from']
    if resume_from:
        net, current_epoch, _resumed_solvers = load_model(net, resume_from, to_use_device,
                                                          third_name=rec_train_options['third_party_name'])
        if _resumed_solvers:
            solvers = _resumed_solvers
        logger.info(f'==> net resume from {resume_from}')
    else:
        logger.info(f'==> net resume from scratch.')

    # ===> loss function
    loss_func = get_loss(cfg['loss'])
    if torch.cuda.is_available and ('cuda' in rec_train_options['device']):
        loss_func = loss_func.to(to_use_device)

    # ===> data loader
    train_loader, eval_loader = get_data_loader(cfg['dataset'])

    # ===> train
    train(net, solvers, schedulers, loss_func, train_loader, eval_loader, to_use_device, rec_train_options,current_epoch, logger)


if __name__ == '__main__':
    main()
