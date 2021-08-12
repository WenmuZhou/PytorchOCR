import subprocess
import os
import numpy as np
import cv2
import torch

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def pse_warpper(kernals, min_area=5):
    '''
    reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
    :param kernals:
    :param min_area:
    :return:
    '''
    from .pse import pse_cpp
    kernal_num = len(kernals)
    if not kernal_num:
        return np.array([]), []
    kernals = np.array(kernals)
    label_num, label = cv2.connectedComponents(kernals[0].astype(np.uint8), connectivity=4)
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)

    pred = pse_cpp(label, kernals, c=kernal_num)

    return np.array(pred), label_values


class pse_postprocess():
    def __init__(self, threshold=0.7311):
        self.threshold = threshold

    def __call__(self, preds, scale):
        """
        在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
        :param preds: 网络输出
        :param scale: 网络的scale
        :param threshold: sigmoid的阈值
        :return: 最后的输出图和文本框
        """
        preds = torch.sigmoid(preds)
        preds = preds.detach().cpu().numpy()

        score = preds[-1].astype(np.float32)
        preds = preds > self.threshold
        # preds = preds * preds[-1] # 使用最大的kernel作为其他小图的mask,不使用的话效果更好
        pred, label_values = pse_warpper(preds, 5)
        bbox_list = []
        for label_value in label_values:
            points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < 800 / (scale * scale):
                continue

            score_i = np.mean(score[pred == label_value])
            if score_i < 0.93:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)
            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        return pred, np.array(bbox_list)
