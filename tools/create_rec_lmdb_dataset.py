# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 15:31
# @Author  : zhoujun

""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, lmdb_save_path, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        data_list  : a list contains img_path\tlabel
        lmdb_save_path : LMDB output path
        checkValid : if true, check the validity of every image
    """
    os.makedirs(lmdb_save_path, exist_ok=True)
    env = lmdb.open(lmdb_save_path, map_size=109951162)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list, desc=f'make dataset, save to {lmdb_save_path}'):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    import pathlib
    label_file = r"path/val.txt"
    lmdb_save_path = r'path/lmdb/eval'
    os.makedirs(lmdb_save_path, exist_ok=True)

    data_list = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'load data from {label_file}'):
            line = line.strip('\n').replace('.jpg ', '.jpg\t').replace('.png ', '.png\t').split('\t')
            if len(line) > 1:
                img_path = pathlib.Path(line[0].strip(' '))
                label = line[1]
                if img_path.exists() and img_path.stat().st_size > 0:
                    data_list.append((str(img_path), label))

    createDataset(data_list, lmdb_save_path)
