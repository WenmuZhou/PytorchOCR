import os
import sys

import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import cv2
import json
import torch

from torchocr.data import create_operators, transform
from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from torchocr.utils.logging import get_logger
from torchocr.utils.visual import draw_det
from torchocr.utils.utility import get_image_file_list
from tools.utility import ArgsParser
from torchocr import Config


def build_det_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)
    return transforms

def main(cfg):
    logger = get_logger()
    global_config = cfg['Global']

    # build model
    model = build_model(cfg['Architecture'])
    load_ckpt(model, cfg)
    model.eval()

    # build post process
    post_process_class = build_post_process(cfg['PostProcess'])

    # create data ops
    transforms = build_det_process(cfg)
    ops = create_operators(transforms, global_config)

    save_res_path = global_config.get('output_dir', 'output')
    os.makedirs(save_res_path, exist_ok=True)
    
    with open(os.path.join(save_res_path, 'predict_det.txt'), "w") as fout:
        for file in get_image_file_list(cfg['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.from_numpy(images)
            with torch.no_grad():
                preds = model(images)
            post_result = post_process_class(preds, [-1, shape_list])

            src_img = cv2.imread(file)
            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": "", "points": np.array(box).tolist()}
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = f'{save_res_path}/det_results_{os.path.basename(file)}'
                    src_img = draw_det(boxes, src_img)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {"transcription": "", "points": np.array(box).tolist()}
                    dt_boxes_json.append(tmp_json)
                save_det_path = f'{save_res_path}/det_results_{os.path.basename(file)}'
                src_img = draw_det(boxes, src_img)
            cv2.imwrite(save_det_path, src_img)
            out_str = f'{file}\t{json.dumps(dt_boxes_json)}'
            fout.write(out_str + '\n')
            logger.info(out_str)
            logger.info("The detected Image saved in {}".format(save_det_path))

    logger.info("success!")


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
