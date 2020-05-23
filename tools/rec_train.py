# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 21:44
# @Author  : xiangjing


import os

import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim import SGD

from tools.rec_train_config import *
from utils import weight_init
from torchocr.networks.architectures.RecModel import RecModel
from dataset.ICDAR15_REC_Dataset import ICDAR15_Rec_Dataset
from torch.nn import CTCLoss


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
    # assert arch_name in {'CRNNMBV3', 'CRNNRes34'}, f'{arch_name} is not developed yet!'
    # module = importlib.import_module(f'torchocr.networks.architectures.RecModel')
    # arch_model = getattr(module, arch_name)
    # return arch_model(**kwargs)
    return RecModel(arch_config)


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
    return [SGD(params, lr=rec_train_options['base_lr'], weight_decay=rec_train_options['weight_decay'])]


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
    for stage in rec_train_options['fine_tune_stage']:
        attr = getattr(net, stage, None)
        for element in attr.parameters():
            yield element
    #     params.append(attr.parameters())
    # return params


def data_loader():
    """
    数据加载，还不完善，需要修改
    :return:
    """
    train_set = ICDAR15_Rec_Dataset()
    test_set = ICDAR15_Rec_Dataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=rec_train_options['batch_size'],
                                               shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,  # one image each batch for testing
                                              shuffle=False, num_workers=4,
                                              pin_memory=True)
    return train_loader, test_loader


def decode(preds):
    pred = []
    for i in range(len(preds)):
        if preds[i] != 0 and ((i == 0) or (i != 0 and preds[i] != preds[i - 1])):
            pred.append(int(preds[i]))
    return pred


def evaluate(net, val_loader, loss_func, max_iter=50):
    logger.info('start val')
    net.eval()
    totalloss = 0.0
    k = 0
    correct_num = 0
    total_num = 0
    val_iter = iter(val_loader)
    max_iter = min(max_iter, len(val_loader))
    for i in range(max_iter):
        k = k + 1
        (data, label) = val_iter.next()
        labels = torch.IntTensor([])
        for j in range(label.size(0)):
            labels = torch.cat((labels, label[j]), 0)
        if torch.cuda.is_available and use_cuda:
            data = data.cuda()
        output = net(data)
        output_size = torch.IntTensor([output.size(0)] * int(output.size(1)))
        label_size = torch.IntTensor([label.size(1)] * int(label.size(0)))
        loss = loss_func(output, labels, output_size, label_size) / label.size(0)
        totalloss += float(loss)
        pred_label = output.max(2)[1]
        pred_label = pred_label.transpose(1, 0).contiguous().view(-1)
        pred = decode(pred_label)
        total_num += len(pred)
        for x, y in zip(pred, labels):
            if int(x) == int(y):
                correct_num += 1
    accuracy = correct_num / float(total_num) * 100
    evaluate_loss = totalloss / k
    logger.info('evaluate loss : %.3f , accuary : %.3f%%' % (evaluate_loss, accuracy))


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

    loss_total = 0.0
    print_interval = rec_train_options['print_interval']
    val_interval = rec_train_options['val_interval']
    num_in_print = 0
    k = 0
    for epoch in range(rec_train_options['epochs']):  # traverse each epoch

        for i, (data, label) in enumerate(train_loader):  # traverse each batch in the epoch
            k = k + 1
            num_in_print = num_in_print + 1
            # put training data, label to device
            if torch.cuda.is_available and use_cuda:
                data = data.to(to_use_device)
                loss_func = loss_func.cuda()

            labels = torch.IntTensor([])
            for j in range(label.size(0)):
                labels = torch.cat((labels, label[j]), 0)

            # forword calculation
            output = net.forward(data)
            output_size = torch.IntTensor([output.size(0)] * int(output.size(1)))
            label_size = torch.IntTensor([label.size(1)] * int(label.size(0)))
            # calculate  loss
            loss = loss_func(output, labels, output_size, label_size) / label.size(0)
            loss_total += float(loss)
            if k % print_interval == 0:
                # display
                logger.info("[%d/%d] || [%d/%d] || Loss:%.3f" % (
                    epoch, rec_train_options['epochs'], i + 1, len(train_loader), loss_total / num_in_print))
                loss_total = 0.0
                num_in_print = 0

            # clear the grad
            solver.zero_grad()

            loss.backward()
            solver.step()
            if k % val_interval == 0:
                # val
                evaluate(net, eval_loader, loss_func)
                net.train()  # train mode
        # 保存ckpt
        if epoch % rec_train_options['ckpt_save_epoch'] ==0:
            torch.save(net.state_dict(),
                       os.path.join(ckpt_dir, 'epoch_'+str(epoch)+'.pth'))


def main():
    logger.info(f'=>train options:\n\t{rec_train_options}')
    to_use_device = torch.device(
        device if torch.cuda.is_available() and use_cuda else 'cpu')
    set_random_seed(SEED, use_cuda, deterministic=True)

    # ===> build network
    net = get_architecture(arc_config)

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
    scheduler = get_lrs(solver[0], rec_train_options['lr_scheduler'])

    # ===> loss function
    loss_func = CTCLoss()

    # ===> data loader
    train_loader, eval_loader = data_loader()

    # ===> train
    train(net, solver[0], scheduler, loss_func, train_loader, eval_loader, to_use_device)


if __name__ == '__main__':
    main()
