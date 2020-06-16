# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 17:06
# @Author  : zhoujun
import torch
import numpy as np
import PIL
import cv2
from torchvision import transforms


class Resize:
    def __init__(self, img_h, img_w, pad=True, **kwargs):
        self.img_h = img_h
        self.img_w = img_w
        self.pad = pad

    def __call__(self, img: np.ndarray):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        img_h = self.img_h
        img_w = self.img_w
        h, w = img.shape[:2]
        ratio_h = self.img_h / h
        new_w = int(w * ratio_h)
        if new_w < img_w and self.pad:
            img = cv2.resize(img, (new_w, img_h))
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            step = np.zeros((img_h, img_w - new_w, img.shape[-1]), dtype=img.dtype)
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (img_w, img_h))
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img


class RecCollateFn:
    def __init__(self, *args, **kwargs):
        self.img_h = kwargs.get('img_h', 32)
        self.img_w = kwargs.get('img_w', 320)
        self.pad = kwargs.get('pad', True)
        self.t = transforms.ToTensor()

    def __call__(self, batch):
        resize_images = []
        resize_image_class = Resize(self.img_h, self.img_w, self.pad)
        labels = []
        for data in batch:
            labels.append(data['label'])
            resize_image = resize_image_class(data['img'])
            resize_image = self.t(resize_image)
            resize_images.append(resize_image)
        resize_images = torch.cat([t.unsqueeze(0) for t in resize_images], 0)
        return {'img':resize_images,'label':labels}
