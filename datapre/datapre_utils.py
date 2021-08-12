#数据格式文件处理的utils

import sys
import os
import cv2
import numpy as np
import random
import time
import json
from PIL import Image


def getlist(path, typ):
    ll = []
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith((typ)):
                ll.append(os.path.join(parent, filename))
        break
    return ll

def readimgs_cv(path_list, flag=None):
    datal = []
    for tmp in path_list:
        datal.append(cv2.imread(tmp, flag))
    return datal

def readimgs_pil(path_list, size=None, flag=None):
    datal = []
    ratio = []
    for tmp in path_list:
        if size is None:
            datal.append(Image.open(tmp).convert('RGBA'))
        else:
            srcimg = Image.open(tmp)
            datal.append(srcimg.resize(size,Image.ANTIALIAS).convert('RGBA'))
            ratio.append((srcimg.size[0]/size[0], srcimg.size[1]/size[1]))
    return datal, ratio


#读取json
def readJson(jsonfile):
    with open(jsonfile,encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData

def writeToJson(filePath,data):
    fb = open(filePath,'w')
    fb.write(json.dumps(data,indent=2)) # ,encoding='utf-8'
    fb.close()

def readjsons(path_list, flag=None):
    datal = []
    for tmp in path_list:
        datal.append(readJson(tmp))
    return datal