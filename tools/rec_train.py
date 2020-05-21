# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing

import importlib
import os
import pickle
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim import SGD

from tools.rec_train_config import *
from utils import weight_init


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


def load_model(_model, resume_from, to_use_device, third_name=None):
    """
    加载预训练模型
    Args:
        _model:  模型
        resume_from: 预训练模型路径
        to_use_device: 设备
        third_name:

    Returns:

    """
    if not third_name:
        if to_use_device.type == 'cpu':
            _model.load_state_dict(torch.load(resume_from, map_location=to_use_device))
        else:
            _model.load_state_dict(torch.load(resume_from))
    elif third_name == 'paddle':
        pass
    return _model


def get_optimizers(params):
    """
    优化器
    Returns:
    """
    return SGD(params, lr=rec_train_options['base_lr'], weight_decay=rec_train_options['weight_decay']),


def get_lrs(optimizer, type='LambdaLR', **kwargs):
    """
    """
    scheduler = None
    if type == 'LambdaLR':
        burn_in, steps = kwargs

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


def get_fine_tune_params(net):
    """
    获取需要优化的参数
    Args:
        net:
    Returns: 需要优化的参数
    """
    params = []
    for stage in rec_train_options['rec_train_options']:
        attr = getattr(net, stage, None)
        params.append(attr.parameters())
    return params


def data_loader():
    """
    数据加载，还不完善，需要修改
    :return:
    """
    train_set = None
    test_set = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=rec_train_options['batch_size'],
                                               shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,  # one image each batch for testing
                                              shuffle=False, num_workers=4,
                                              pin_memory=True)
    return train_loader, test_loader


def train(net, solver, scheduler, loss_func, train_loader, eval_loader, to_use_device):
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
    Returns:
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info('==> Training...')
    net.train()  # train mode
    best_acc = 0.0
    best_epoch = None
    err_dict = {}
    for t in range(rec_train_options['epochs']):  # traverse each epoch
        epoch_loss = []
        num_correct = 0
        num_total = 0
        for rect_info in train_loader:  # traverse each batch in the epoch
            # put training data, label to device
            for rect in rect_info:
                pass

        #     data, label = data.to(to_use_device), label.to(to_use_device)
        #     # clear the grad
        #     solver.zero_grad()
        #     # forword calculation
        #     output = net.forward(data)
        #     # calculate each attribute loss
        #     label = label.long()
        #     loss_rec = loss_func(output[:, :3], label[:, 0])
        #
        #     # statistics of each epoch loss
        #     epoch_loss.append(loss_rec)
        #
        #     # statistics of sample number
        #     num_total += label.size(0)
        #
        #     # statistics of accuracy
        #     # pred = get_predict(output)
        #     # label = label.cpu().long()
        #     # num_correct += count_correct(pred, label)
        #
        #     # backward calculation according to loss
        #     loss_rec.backward()
        #     solver.step()
        # print(solver.state_dict()['param_groups'][0]['lr'])
        #
        # # calculate training accuray
        # train_acc = 100.0 * float(num_correct) / float(num_total)

        # calculate accuracy of test set every epoch
        # eval_acc = evaluate_accuracy(net, eval_loader, is_draw=False)

        # schedule the learning rate according to test acc
        # scheduler.step(test_acc)

    #     # 保存精度最好的epoch
    #     if eval_acc > best_acc:
    #         best_acc = eval_acc
    #         best_epoch = t + 1
    #
    #         # dump model to disk
    #         model_save_name = 'epoch_' + \
    #                           str(t + latest_model_id + 1) + '.pth'
    #         torch.save(net.state_dict(),
    #                    os.path.join(paths['ckpt_dir'], model_save_name))
    #         logger.info('<= {} saved.'.format(model_save_name))
    #     logger.info('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
    #                 (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
    #
    #     err_dict_path = './err_dict.pkl'
    #     pickle.dump(err_dict, open(err_dict_path, 'wb'))
    #     print('=> err_dict dumped @ %s' % err_dict_path)
    #     err_dict = {}  # reset err dict
    #
    # logger.info('=> Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))


def main():
    logger.info(f'=>train options:\n\t{rec_train_options}')
    to_use_device = torch.device(
        device if torch.cuda.is_available() and use_cuda else 'cpu')
    set_random_seed(SEED, use_cuda, deterministic=True)

    # ===> build network
    net = get_architecture(architecture, **architecture_config)

    # ===> 模型初始化
    net.apply(weight_init)
    net.to(to_use_device)

    # ===> whether to resume from checkpoint
    if resume_from:
        net = load_model(net, resume_from, to_use_device)
        logger.info(f'==> net resume from {resume_from}')
    else:
        logger.info(f'==> net resume from scratch.')

    # ===> get fine tune layers
    params = get_fine_tune_params(net)
    # ===> solver and lr scheduler
    solver = get_optimizers(params)
    scheduler = get_lrs(solver, rec_train_options['lr_scheduler'])

    # ===> loss function
    loss_func = None

    # ===> data loader
    train_loader, eval_loader = data_loader()

    # ===> train
    train(net, solver, scheduler, loss_func, train_loader, eval_loader, to_use_device)


if __name__ == '__main__':
    main()
