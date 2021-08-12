# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 10:53
# @Author  : zhoujun
import os
import cv2
import json
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torchocr.datasets.det_modules import *


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


class JsonDataset(Dataset):
    """
    from https://github.com/WenmuZhou/OCR_DataSet/blob/master/dataset/det.py
    """
    def __init__(self, config):
        assert config.img_mode in ['RGB', 'BRG', 'GRAY']
        self.ignore_tags = config.ignore_tags
        # 加载字符级标注
        self.load_char_annotation = False

        self.data_list = self.load_data(config.file)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(item_keys)
        self.img_mode = config.img_mode
        self.filter_keys = config.filter_keys
        self._init_pre_processes(config.pre_processes)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, path: str) -> list:
        """
        从json文件中读取出 文本行的坐标和gt，字符的坐标和gt
        :params path: 存储数据的文件夹或者文件
        return a dict ,包含了，'img_path','img_name','text_polys','texts','ignore_tags'
        """
        data_list = []
        content = load_json(path)
        for gt in tqdm(content['data_list'], desc='read file {}'.format(path)):
            img_path = os.path.join(content['data_root'], gt['img_name'])
            polygons = []
            texts = []
            illegibility_list = []
            language_list = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0 or len(annotation['text']) == 0:
                    continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                illegibility_list.append(annotation['illegibility'])
                language_list.append(annotation['language'])
                if self.load_char_annotation:
                    for char_annotation in annotation['chars']:
                        if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                            continue
                        polygons.append(char_annotation['polygon'])
                        texts.append(char_annotation['char'])
                        illegibility_list.append(char_annotation['illegibility'])
                        language_list.append(char_annotation['language'])
            data_list.append({'img_path': img_path, 'img_name': gt['img_name'], 'text_polys': np.array(polygons),
                              'texts': texts, 'ignore_tags': illegibility_list})
        return data_list

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        # try:
        data = copy.deepcopy(self.data_list[index])
        im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        data['img'] = im
        data['shape'] = [im.shape[0], im.shape[1]]
        data = self.apply_pre_processes(data)

        if self.transform:
            data['img'] = self.transform(data['img'])
        data['text_polys'] = data['text_polys']
        if len(self.filter_keys):
            data_dict = {}
            for k, v in data.items():
                if k not in self.filter_keys:
                    data_dict[k] = v
            return data_dict
        else:
            return data
        # except:
        #     return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from config.det_train_db_config import config
    from torchocr.utils import show_img, draw_bbox

    from matplotlib import pyplot as plt
    dataset = JsonDataset(config.dataset.train.dataset)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(tqdm(train_loader)):
        img = data['img']
        shrink_label = data['shrink_map']
        threshold_label = data['threshold_map']

        print(threshold_label.shape, threshold_label.shape, img.shape)
        show_img(img[0].numpy().transpose(1, 2, 0), title='img')
        show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label')
        show_img((threshold_label[0].to(torch.float)).numpy(), title='threshold_label')
        img = draw_bbox(img[0].numpy().transpose(1, 2, 0), np.array(data['text_polys']))
        show_img(img, title='draw_bbox')
        plt.show()

        pass
