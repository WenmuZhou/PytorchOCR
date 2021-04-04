# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 14:16
# @Author  : zhoujun
import PIL
import numpy as np
import torch
from torchvision import transforms

__all__ = ['DetCollectFN']


class DetCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                    if isinstance(v, np.ndarray):
                        v = torch.tensor(v)
                    if isinstance(v, PIL.Image.Image):
                        v = transforms.ToTensor()(v)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict
