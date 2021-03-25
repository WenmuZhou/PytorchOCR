"""
@Author: Jeffery Sheng (Zhenfei Sheng)
@Time:   2020/5/21 19:44
@File:   RecDataSet.py
"""
import six
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchocr.utils.CreateRecAug import cv2pil, pil2cv, RandomBrightness, RandomContrast, \
    RandomLine, RandomSharpness, Compress, Rotate, \
    Blur, MotionBlur, Salt, AdjustResolution


class RecTextLineDataset(Dataset):
    def __init__(self, config):
        """
        文本行 DataSet, 用于处理标注格式为 `img_path\tlabel` 的标注格式

        :param config: 相关配置，一般为 config['dataset']['train']['dataset] or config['dataset']['eval']['dataset]
                其主要应包含如下字段： file: 标注文件路径
                                    input_h: 图片的目标高
                                    mean: 归一化均值
                                    std: 归一化方差
                                    augmentation: 使用使用数据增强
        :return None
        """
        self.augmentation = config.augmentation
        self.process = RecDataProcess(config)
        with open(config.alphabet, 'r', encoding='utf-8') as file:
            alphabet = ''.join([s.strip('\n') for s in file.readlines()])
        alphabet += ' '
        self.str2idx = {c: i for i, c in enumerate(alphabet)}
        self.labels = []
        with open(config.file, 'r', encoding='utf-8') as f_reader:
            for m_line in f_reader.readlines():
                params = m_line.split('\t')
                if len(params) == 2:
                    m_image_name, m_gt_text = params
                    if True in [c not in self.str2idx for c in m_gt_text]:
                        continue
                    self.labels.append((m_image_name, m_gt_text))

    def _find_max_length(self):
        return max({len(_[1]) for _ in self.labels})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get img_path and trans
        img_path, trans = self.labels[index]
        # read img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # do aug
        if self.augmentation:
            img = pil2cv(self.process.aug_img(cv2pil(img)))
        return {'img': img, 'label': trans}


class RecLmdbDataset(Dataset):
    def __init__(self, config):
        """
        Lmdb DataSet, 用于处理转换为 lmdb 文件后的数据集

        :param config: 相关配置，一般为 config['dataset']['train']['dataset] or config['dataset']['eval']['dataset]
                其主要应包含如下字段： file: 标注文件路径
                                    input_h: 图片的目标高
                                    mean: 归一化均值
                                    std: 归一化方差
                                    augmentation: 使用使用数据增强
        :return None
        """
        import lmdb, sys
        self.env = lmdb.open(config.file, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (config.file))
            sys.exit(0)

        self.augmentation = config.augmentation
        self.process = RecDataProcess(config)
        self.filtered_index_list = []
        self.labels = []
        with open(config.alphabet, 'r', encoding='utf-8') as file:
            alphabet = ''.join([s.strip('\n') for s in file.readlines()])
        alphabet += ' '
        self.str2idx = {c: i for i, c in enumerate(alphabet)}
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')
                # todo 添加 过滤最长
                # if len(label) > config.max_len:
                #     # print(f'The length of the label is longer than max_length: length
                #     # {len(label)}, {label} in dataset {self.root}')
                #     continue
                if True in [c not in self.str2idx for c in label]:
                    continue
                # By default, images containing characters which are not in opt.character are filtered.
                # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                self.labels.append(label)
                self.filtered_index_list.append(index)

    def _find_max_length(self):
        return max({len(_) for _ in self.labels})

    def __getitem__(self, index):
        index = self.filtered_index_list[index]
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')  # for color image
            # We only train and evaluate on alphanumerics (or pre-defined character set in rec_train.py)
            img = np.array(img)
            if self.augmentation:
                img = pil2cv(self.process.aug_img(cv2pil(img)))
        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.filtered_index_list)


class RecDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers, **kwargs):
        """
        自定义 DataLoader, 主要实现数据集的按长度划分，将长度相近的放在一个 batch

        :param dataset: 继承自 torch.utils.data.DataSet的类对象
        :param batch_size: 一个 batch 的图片数量
        :param shuffle: 是否打乱数据集
        :param num_workers: 后台进程数
        :param kwargs: **
        """
        self.dataset = dataset
        self.process = dataset.process
        self.len_thresh = self.dataset._find_max_length() // 2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
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
        batch = {'img': [], 'label': []}
        # img tensor current shape: B,H,W,C
        all_same_height_images = [self.process.resize_with_specific_height(_['img'][0].numpy()) for _ in batch_data]
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # make sure max_img_w is integral multiple of 8
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        for i in range(len(batch_data)):
            _label = batch_data[i]['label'][0]
            img = self.process.normalize_img(self.process.width_pad_img(all_same_height_images[i], max_img_w))
            img = img.transpose([2, 0, 1])
            batch['img'].append(torch.tensor(img, dtype=torch.float))
            batch['label'].append(_label)
        batch['img'] = torch.stack(batch['img'])
        return batch

    def build(self):
        self.dataiter = DataLoader(self.dataset, batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers).__iter__()

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
                if len(temp['label'][0]) <= self.len_thresh:
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
        """
        文本是被数据增广类

        :param config: 配置，主要用到的字段有 input_h, mean, std
        """
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
        to_return_img = np.ones([_height, _target_width, _channels], dtype=_img.dtype) * _pad_value
        to_return_img[:_height, :_width, :] = _img
        return to_return_img
