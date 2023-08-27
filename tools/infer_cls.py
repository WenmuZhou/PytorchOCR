# -*- coding: utf-8 -*-
# @Time    : 2023/8/27 1:53
# @Author  : zhoujun
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
import json
import numpy as np
import torch

from torchocr.data import create_operators, transform
from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from torchocr.utils.utility import get_image_file_list
from torchocr.utils.logging import get_logger
from tools.utility import update_rec_head_out_channels, ArgsParser
from torchocr import Config

def build_cls_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image']
        elif op_name == "SSLRotateResize":
            op[op_name]["mode"] = "test"
        transforms.append(op)
    return transforms

def main(cfg):
    logger = get_logger()
    global_config = cfg['Global']

    # build post process
    post_process_class = build_post_process(cfg['PostProcess'])

    update_rec_head_out_channels(cfg, post_process_class)
    model = build_model(cfg['Architecture'])
    load_ckpt(model, cfg)
    model.eval()

    # create data ops
    transforms = build_cls_process(cfg)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    for file in get_image_file_list(cfg['Global']['infer_img']):
        logger.info("infer_img: {}".format(file))
        with open(file, 'rb') as f:
            img = f.read()
            data = {'image': img}
        batch = transform(data, ops)

        images = np.expand_dims(batch[0], axis=0)
        images = torch.from_numpy(images)
        preds = model(images)
        post_result = post_process_class(preds)
        for rec_result in post_result:
            logger.info('\t result: {}'.format(rec_result))
    logger.info("success!")


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
