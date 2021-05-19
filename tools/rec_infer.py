# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))

import numpy as np

sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter


class RecInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
        self.converter = CTCLabelConverter(cfg['dataset']['alphabet'])

    def predict(self, imgs):
        # 预处理根据训练来
        if not isinstance(imgs,list):
            imgs = [imgs]
        imgs = [self.process.normalize_img(self.process.resize_with_specific_height(img)) for img in imgs]
        imgs = [self.process.width_pad_img(img, 256) for img in imgs]
        imgs = np.stack(imgs)
        tensor = torch.from_numpy(imgs.transpose([0,3, 1, 2])).float()
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            out = out.softmax(dim=2)
        out = out.cpu().numpy()
        txts = [self.converter.decode(np.expand_dims(txt, 0)) for txt in out]
        return txts


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import cv2

    args = init_args()
    img = cv2.imread(args.img_path)
    model = RecInfer(args.model_path)
    out = model.predict(img)
    print(out)
