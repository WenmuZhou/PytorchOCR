'''
@Author: Jeffery Sheng (Zhenfei Sheng)
@Time:   2020/5/21 19:44
@File:   ICDAR15RecDataset.py
'''

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import StrLabelConverter
from util_scripts.CreateRecAug import cv2pil, pil2cv, RandomBrightness, RandomContrast, \
                                      RandomLine, RandomSharpness, Compress, Rotate, \
                                      Blur, MotionBlur, Salt, AdjustResolution


class icdar15RecDataset(Dataset):
    def __init__(self, config):
        '''
        :param config: dataset config, need data_dir, input_h, mean, std,
        mode:'train' or 'val', augmentation: True or False,
        batch_size, shuffle, num_workers
        '''
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
            self.labels = [{line.split('\t')[0]: line.split('\t')[-1][:-1]} for line in file.readlines()]

        print(f'load {self.__len__()} images.')

    def _findmaxlength(self):
        return max({len(list(d.values())[0]) for d in self.labels})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get img_path
        img_name = list(self.labels[index].keys())[0]
        img_path = os.path.join(self.data_dir, f'images/{img_name}')

        # get trans
        trans = list(self.labels[index].values())[0]
        # convert to label
        label, length = self.converter.encode(trans)
        # read img
        img = cv2.imread(img_path)
        # do aug
        if self.augmentation:
            img = pil2cv(RecDataProcess(self.config).aug_img(cv2pil(img)))
        # to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, label, length

class RecDataLoader:
    def __init__(self, dataset: Dataset, config):
        self.dataset = dataset
        self.process = RecDataProcess(config)
        self.len_thresh = self.dataset._findmaxlength() // 2
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_workers = config.num_workers
        self.iteration = 0
        self.dataiter = None
        self.queue_1 = list()
        self.queue_2 = list()

    def __len__(self):
        return len(self.dataset)//self.batch_size if len(self.dataset) % self.batch_size == 0 \
        else len(self.dataset)//self.batch_size + 1

    def __iter__(self):
        return self

    def pack(self, batch_data):
        batch = [[], [], []]
        max_length = max({it[2].item() for it in batch_data})
        # img tensor current shape: C, H, W
        max_img_w = max({it[0].shape[-1] for it in batch_data})
        # make sure max_img_w is integral multiple of 8
        max_img_w = max_img_w + (8 - max_img_w % 8) if max_img_w % 8 != 0 else max_img_w
        for i in range(len(batch_data)):
            _img, _label, _length = batch_data[i]
            # trans to np array, roll back axis
            _img = _img.numpy().transpose([1, 2, 0])
            img = self.process.resize_normalize(_img, max_img_w)
            label = _label.tolist()[0] + [0] * (max_length - len(_label.tolist()[0]))
            batch[0].append(torch.FloatTensor(img))
            batch[1].append(torch.IntTensor(label))
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

    def resize_normalize(self, img, input_w):
        # to resize in proportion
        # width should change with the max length
        img = cv2.resize(img, (0, 0), fx=input_w / img.shape[1],
                         fy=self.config.input_h / img.shape[0], interpolation=cv2.INTER_CUBIC)
        # add third axis
        img = np.reshape(img, (self.config.input_h, input_w, 1))
        # to float32
        img = img.astype(np.float32)
        # normalize
        img = (img/255. - self.config.mean) / self.config.std
        # roll axis
        img = img.transpose([2, 0, 1])
        return img