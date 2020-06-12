"""
@Author: Jeffery Sheng (Zhenfei Sheng)
@Time:   2020/5/21 19:44
@File:   ICDAR15RecDataset.py
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import StrLabelConverter
from util_scripts.CreateRecAug import cv2pil, pil2cv, RandomBrightness, RandomContrast, \
    RandomLine, RandomSharpness, Compress, Rotate, \
    Blur, MotionBlur, Salt, AdjustResolution


class ICDAR15RecDataset(Dataset):
    def __init__(self, config):
        """
        :param config: dataset config, need data_dir, input_h, mean, std,
        mode:'train' or 'val', augmentation: True or False,
        batch_size, shuffle, num_workers
        """
        self.config = config
        self.data_dir = config.data_dir
        self.input_h = config.input_h
        self.mode = config.mode
        self.alphabet = config.alphabet
        self.mean = np.array(config.mean, dtype=np.float32)
        self.std = np.array(config.std, dtype=np.float32)
        self.augmentation = config.augmentation

        # get alphabet
        with open(self.alphabet, 'r') as file:
            alphabet = ''.join([s.strip('\n') for s in file.readlines()])
        # get converter
        self.converter = StrLabelConverter(alphabet, False)

        # build path of train.txt of val.txt
        gt_path = os.path.join(self.data_dir, f'{self.mode}.txt')
        with open(gt_path, 'r', encoding='utf-8') as file:
            # build {img_path: trans}
            self.labels = []
            for m_line in file:
                m_image_name, m_gt_text = m_line.strip().split('\t')
                self.labels.append((m_image_name, m_gt_text))

        print(f'load {self.__len__()} images.')

    def _find_max_length(self):
        return max({len(_[1]) for _ in self.labels})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get img_path and trans
        img_name, trans = self.labels[index]
        img_path = os.path.join(self.data_dir, 'image', img_name)

        # convert to label
        label, length = self.converter.encode(trans)
        # read img
        img = cv2.imread(img_path)
        # do aug
        if self.augmentation:
            img = pil2cv(RecDataProcess(self.config).aug_img(cv2pil(img)))
        return img, label, length


class RecDataLoader:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.process = RecDataProcess(config)
        self.len_thresh = self.dataset._find_max_length() // 2
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_workers = config.num_workers
        self.iteration = 0
        self.dataiter = None
        self.queue_1 = list()
        self.queue_2 = list()

    def __len__(self):
        return len(self.dataset) // self.batch_size if len(self.dataset) % self.batch_size == 0 \
            else len(self.dataset) // self.batch_size + 1

    def __iter__(self):
        return self

    def pack(self, batch_data):
        batch = [[], [], []]
        max_length = max({it[2].item() for it in batch_data})
        # img tensor current shape: B,H,W,C
        all_same_height_images = [self.process.resize_with_specific_height(_[0][0].numpy()) for _ in batch_data]
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # make sure max_img_w is integral multiple of 8
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        for i in range(len(batch_data)):
            _, _label, _length = batch_data[i]
            img = self.process.normalize_img(self.process.width_pad_img(all_same_height_images[i], max_img_w))
            img = img.transpose([2, 0, 1])
            label = torch.zeros([max_length])
            label[:_length.item()] = _label
            batch[0].append(torch.FloatTensor(img))
            batch[1].append(label.to(dtype=torch.int32))
            batch[2].append(torch.IntTensor([max_length]))

        return [torch.stack(batch[0]), torch.stack(batch[1]), torch.cat(batch[2])]

    def build(self):
        self.dataiter = DataLoader(self.dataset, batch_size=1,
                                   shuffle=self.shuffle, num_workers=self.num_workers).__iter__()

    def __next__(self):
        if self.dataiter == None:
            self.build()
        if self.iteration == len(self.dataset) and len(self.queue_2):
            batch_data = self.queue_2
            self.queue_2 = list()
            return self.pack(batch_data)
        if not len(self.queue_2) and not len(self.queue_1) and self.iteration == len(self.dataset):
            self.iteration = 0
            self.dataiter = None
            raise StopIteration
        # start iteration
        try:
            while True:
                # get data from origin dataloader
                temp = self.dataiter.__next__()
                self.iteration += 1
                # to different queue
                if temp[2].item() <= self.len_thresh:
                    self.queue_1.append(temp)
                else:
                    self.queue_2.append(temp)

                # to store batch data
                batch_data = None
                # queue_1 full, push to batch_data
                if len(self.queue_1) == self.batch_size:
                    batch_data = self.queue_1
                    self.queue_1 = list()
                # or queue_2 full, push to batch_data
                elif len(self.queue_2) == self.batch_size:
                    batch_data = self.queue_2
                    self.queue_2 = list()

                # start to process batch
                if batch_data is not None:
                    return self.pack(batch_data)
        # deal with last batch
        except StopIteration:
            batch_data = self.queue_1
            self.queue_1 = list()
            return self.pack(batch_data)


class RecDataProcess:
    def __init__(self, config):
        self.config = config
        self.random_contrast = RandomContrast(probability=0.3)
        self.random_brightness = RandomBrightness(probability=0.3)
        self.random_sharpness = RandomSharpness(probability=0.3)
        self.compress = Compress(probability=0.3)
        self.rotate = Rotate(probability=0.5)
        self.blur = Blur(probability=0.3)
        self.motion_blur = MotionBlur(probability=0.3)
        self.salt = Salt(probability=0.3)
        self.adjust_resolution = AdjustResolution(probability=0.3)
        self.random_line = RandomLine(probability=0.3)
        self.random_contrast.setparam()
        self.random_brightness.setparam()
        self.random_sharpness.setparam()
        self.compress.setparam()
        self.rotate.setparam()
        self.blur.setparam()
        self.motion_blur.setparam()
        self.salt.setparam()
        self.adjust_resolution.setparam()

    def aug_img(self, img):
        img = self.random_contrast.process(img)
        img = self.random_brightness.process(img)
        img = self.random_sharpness.process(img)
        img = self.random_line.process(img)

        if img.size[1] >= 32:
            img = self.compress.process(img)
            img = self.adjust_resolution.process(img)
            img = self.motion_blur.process(img)
            img = self.blur.process(img)
        img = self.rotate.process(img)
        img = self.salt.process(img)
        return img

    def resize_with_specific_height(self, _img):
        """
        将图像resize到指定高度
        :param _img:    待resize的图像
        :return:    resize完成的图像
        """
        resize_ratio = self.config.input_h / _img.shape[0]
        return cv2.resize(_img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)

    def normalize_img(self, _img):
        """
        根据配置的均值和标准差进行归一化
        :param _img:    待归一化的图像
        :return:    归一化后的图像
        """
        return (_img.astype(np.float32) / 255 - self.config.mean) / self.config.std

    def width_pad_img(self, _img, _target_width, _pad_value=0):
        """
        将图像进行高度不变，宽度的调整的pad
        :param _img:    待pad的图像
        :param _target_width:   目标宽度
        :param _pad_value:  pad的值
        :return:    pad完成后的图像
        """
        _height, _width, _channels = _img.shape
        to_return_img = np.ones([_height, _target_width, _channels]) * _pad_value
        to_return_img[:_height, :_width, :] = _img
        return to_return_img
