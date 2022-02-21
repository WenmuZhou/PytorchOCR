# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 8:56
# @Author  : zhoujun
import os
import torch


def load_checkpoint(_model, resume_from, to_use_device, _optimizers=None, third_name=None):
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
    global_state = {}
    if not third_name:
        state = torch.load(resume_from, map_location=to_use_device)
        _model.load_state_dict({'module.' + k: v for k, v in state['state_dict'].items()}, strict=True)
        # _model.load_state_dict(state['state_dict'])
        if 'optimizer' in state and _optimizers is not None:
            _optimizers.load_state_dict(state['optimizer'])
        if 'global_state' in state:
            global_state = state['global_state']

    return _model, _optimizers, global_state


def save_checkpoint(checkpoint_path, model, _optimizers, logger, cfg, **kwargs):
    # if isinstance(model, torch.nn.DataParallel()):
    #     mode_state_dict = model.module.state_dict()
    # else:
    mode_state_dict = model.module.state_dict()
    state = {'state_dict': mode_state_dict,
             'optimizer': _optimizers.state_dict(),
             'cfg': cfg}
    state.update(kwargs)
    torch.save(state, checkpoint_path)
    logger.info('models saved to %s' % checkpoint_path)


def save_checkpoint_logic(total_loss, total_num, min_loss, net, solver, epoch, rec_train_options, logger):
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
