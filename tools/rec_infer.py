# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import torch
from torch import nn
from torchocr.networks import build_model
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter


class RecInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        from config.rec_train_config import config
        self.model = build_model(config['model'])
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(ckpt['state_dict'])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
        self.converter = CTCLabelConverter(cfg['dataset']['alphabet'])

    def predict(self, img):
        # 预处理根据训练来
        img = self.process.resize_with_specific_height(img)
        # img = self.process.width_pad_img(img, 120)
        img = self.process.normalize_img(img)
        tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tensor = tensor.unsqueeze(dim=0)
        out = self.model(tensor)
        txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
        return txt

if __name__ == '__main__':
    import cv2
    model_path = 'out_dir/checkpoint/best.pth'
    img_path = r'E:\zj\dataset\test_crnn_digit\val\11_song5_0_3_w.jpg'
    img = cv2.imread(img_path)
    model = RecInfer(model_path)
    out = model.predict(img)
    print(out)