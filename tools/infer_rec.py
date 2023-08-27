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

def build_rec_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if cfg['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif cfg['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif cfg['Architecture']['algorithm'] == "RobustScanner":
                op[op_name][
                    'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
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
    transforms = build_rec_process(cfg)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = global_config.get('output_dir', 'output')
    os.makedirs(save_res_path, exist_ok=True)

    with open(os.path.join(save_res_path, 'predict_res.txt'), "w") as fout:
        for file in get_image_file_list(global_config['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            others = None
            if cfg['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    torch.from_numpy(encoder_word_pos_list),
                    torch.from_numpy(gsrm_word_pos_list),
                    torch.from_numpy(gsrm_slf_attn_bias1_list),
                    torch.from_numpy(gsrm_slf_attn_bias2_list)
                ]
            elif cfg['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                others = [torch.from_numpy(valid_ratio)]
            elif cfg['Architecture']['algorithm'] == "RobustScanner":
                valid_ratio = np.expand_dims(batch[1], axis=0)
                word_positons = np.expand_dims(batch[2], axis=0)
                others = [
                    torch.from_numpy(valid_ratio),
                    torch.from_numpy(word_positons),
                ]
            images = np.expand_dims(batch[0], axis=0)
            images = torch.from_numpy(images)
            preds = model(images, others)
            post_result = post_process_class(preds)
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            elif isinstance(post_result, list) and isinstance(post_result[0],
                                                              int):
                # for RFLearning CNT branch
                info = str(post_result[0])
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0] + "\t" + str(post_result[0][1])

            logger.info(f"{file}\t result: {info}")
            fout.write(f"{file}\t result: {info}\n")
    logger.info("success!")


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
