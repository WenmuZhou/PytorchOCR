# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 17:01
# @Author  : zhoujun
import time
import shutil
import traceback
import torch
from tqdm import tqdm
from torchocr.utils import save_checkpoint, AverageMeter


def evaluate(net, val_loader, loss_func, to_use_device, logger, converter, metric):
    logger.info('start evaluate')
    net.eval()
    nums = 0
    result_dict = {'eval_loss': 0., 'eval_acc': 0., 'eval_ned': 0.}
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            targets, targets_lengths = converter.encode(batch_data['label'])
            batch_data['targets'] = targets
            batch_data['targets_lengths'] = targets_lengths
            output = net.forward(batch_data['img'].to(to_use_device))
            loss = loss_func(output, batch_data)
            result_dict['eval_loss'] += loss['loss'].item()
            nums += batch_data['img'].shape[0]
            acc_dict = metric(output, batch_data['label'])
            result_dict['eval_acc'] = acc_dict['n_correct']
            result_dict['norm_edit_dis'] = acc_dict['norm_edit_dis']

    for key, val in result_dict.items():
        result_dict[key] = result_dict[key] / nums
        logger.info('{}:{}'.format(key, result_dict[key]))
    result_dict['norm_edit_dis'] = 1 - result_dict['norm_edit_dis']
    net.train()
    return result_dict


def train(net, optimizer, scheduler, loss_func, train_loader, eval_loader, to_use_device,
          cfg, _epoch, logger):
    """
    Returns:
    :param net: 模型
    :param optimizer: 优化器
    :param scheduler: 学习率更新
    :param loss_func: loss函数
    :param train_loader: 训练数据集dataloader
    :param eval_loader: 验证数据集dataloader
    :param to_use_device:
    :param train_options:
    :param _epoch:
    :param logger:
    """
    from torchocr.metrics import RecMetric
    from torchocr.utils import CTCLabelConverter
    with open(cfg.dataset.alphabet, 'r', encoding='utf-8') as file:
        cfg.dataset.alphabet = ''.join([s.strip('\n') for s in file.readlines()])
    converter = CTCLabelConverter(cfg.dataset.alphabet)
    train_options = cfg.train_options
    metric = RecMetric(converter)
    # ===>
    logger.info('Training...')
    # ===> print loss信息的参数
    average = AverageMeter()
    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_loader.dataset.__len__(), all_step))
    logger.info('eval dataset has {} samples,{} in dataloader'.format(eval_loader.dataset.__len__(), len(eval_loader)))
    best_model = {'best_acc': 0, 'eval_loss': 0, 'model_path': '', 'eval_acc': 0., 'eval_ned': 0.}
    # 开始训练
    try:
        start = time.time()
        for epoch in range(_epoch, train_options['epochs']):  # traverse each epoch
            current_lr = scheduler.get_last_lr()[0]
            net.train()  # train mode
            for i, batch_data in enumerate(train_loader):  # traverse each batch in the epoch
                cur_batch_size = batch_data['img'].shape[0]
                targets, targets_lengths = converter.encode(batch_data['label'])
                batch_data['targets'] = targets
                batch_data['targets_lengths'] = targets_lengths

                optimizer.zero_grad()
                output = net.forward(batch_data['img'].to(to_use_device))
                loss_dict = loss_func(output, batch_data)
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
                # statistic loss for print
                average['loss'] = (loss_dict['loss'].item(), 1)
                acc_dict = metric(output, batch_data['label'])
                average['acc'] = (acc_dict['n_correct'], cur_batch_size)
                average['norm_edit_dis'] = (acc_dict['norm_edit_dis'], cur_batch_size)
                if (i + 1) % train_options['print_interval'] == 0:
                    interval_batch_time = time.time() - start
                    logger.info(f"[{epoch}/{train_options['epochs']}] - "
                                f"[{i + 1}/{all_step}] -"
                                f"lr:{current_lr} - "
                                f"loss:{average['loss']:.4f} - "
                                f"acc:{average['acc']:.4f} - "
                                f"norm_edit_dis:{1 - average['norm_edit_dis']:.4f} - "
                                f"time:{interval_batch_time:.4f}")
                    start = time.time()
                if (i + 1) >= train_options['val_interval'] and (i + 1) % train_options['val_interval'] == 0:
                    # val
                    eval_dict = evaluate(net, eval_loader, loss_func, to_use_device, logger, converter, metric)
                    if train_options['ckpt_save_type'] == 'HighestAcc':
                        net_save_path = f"{train_options['checkpoint_save_dir']}/latest.pth"
                        save_checkpoint(net_save_path, net, optimizer, epoch, logger, cfg)
                        if eval_dict['eval_acc'] > best_model['eval_acc']:
                            best_model.update(eval_dict)
                            best_model['models'] = net_save_path
                            shutil.copy(net_save_path, net_save_path.replace('latest', 'best'))
                    elif train_options['ckpt_save_type'] == 'FixedEpochStep' and epoch % train_options['ckpt_save_epoch'] == 0:
                        net_save_path = f"{train_options['checkpoint_save_dir']}/{epoch}.pth"
                        save_checkpoint(net_save_path, net, optimizer, epoch, logger, cfg)
            scheduler.step()
    except KeyboardInterrupt:
        import os
        save_checkpoint(os.path.join(train_options['checkpoint_save_dir'], 'final_' + str(epoch) + '.pth'), net,
                        optimizer, epoch, logger, cfg)
    except:
        error_msg = traceback.format_exc(limit=1)
        logger.error(error_msg)
    finally:
        for k, v in best_model.items():
            logger.info(f'{k}: {v}')
