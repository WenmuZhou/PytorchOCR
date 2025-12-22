# -*- coding: utf-8 -*-
# @Time    : 2023/8/27 10:02
# @Author  : zhoujun
import os

import torch

from torchocr.utils.logging import get_logger


def save_ckpt(model, cfg, optimizer, lr_scheduler, epoch, global_step, metrics, is_best=False, logger=None):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
    """
    if logger is None:
        logger = get_logger()
    if is_best:
        save_path = os.path.join(cfg['Global']['output_dir'], 'best.pth')
    else:
        save_path = os.path.join(cfg['Global']['output_dir'], 'latest.pth')
    state_dict = model.module.state_dict() if cfg['Global']['distributed'] else model.state_dict()
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'config': cfg,
        'metrics': metrics
    }
    torch.save(state, save_path)
    logger.info(f'save ckpt to {save_path}')


def load_ckpt(model, cfg, optimizer=None, lr_scheduler=None, logger=None):
    """
    Resume from saved checkpoints
    :param checkpoint_path: Checkpoint path to be resumed
    """
    if logger is None:
        logger = get_logger()
    checkpoints = cfg['Global'].get('checkpoints')
    pretrained_model = cfg['Global'].get('pretrained_model')

    status = {}
    if checkpoints and os.path.exists(checkpoints):
        checkpoint = torch.load(checkpoints, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint.get('epoch', 'N/A')
        logger.info(f"resume from checkpoint: {checkpoints} (epoch {epoch})")
        # logger.info(f"resume from checkpoint: {checkpoints} (epoch {checkpoint['epoch']})")

        # status['global_step'] = checkpoint['global_step']
        # status['epoch'] = checkpoint['epoch'] + 1
        # status['metrics'] = checkpoint['metrics']
        status['epoch'] = checkpoint.get('epoch', 0)
        status['global_step'] = checkpoint.get('global_step', 0)
        status['metrics'] = checkpoint.get('metrics', {})
    elif pretrained_model and os.path.exists(pretrained_model):
        load_pretrained_params(model, pretrained_model)
        logger.info(f"finetune from checkpoint: {pretrained_model}")
    else:
        logger.info("train from scratch")
    return status


def load_pretrained_params(model, pretrained_model):
    logger = get_logger()
    loaded_state_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))['state_dict']
    current_model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        if k not in current_model_dict.keys():
            logger.info(f"ignore loading parameter: {k}, because it is not in current model")
            continue

        if current_model_dict[k].size() != v.size():
            logger.info(f"ignore loading parameter: {k}, because of size mismatch, current size: {current_model_dict[k].size()}, pretrained size: {v.size()}")
            continue
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
