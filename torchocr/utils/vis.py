# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 10:34
# @Author  : zhoujun
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def get_font_file():
    searchs = ["./doc/田氏颜体大字库2.0.ttf", "../doc/田氏颜体大字库2.0.ttf"]
    for path in searchs:
        if os.path.exists(path):
            return path
    assert False,"can't find 田氏颜体大字库2.0.ttf"


def draw_ocr_box_txt(image, boxes, txts = None, pos="horizontal"):
    if isinstance(image,np.ndarray):
        image = Image.fromarray(image)
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random
    # 每次使用相同的随机种子 ，可以保证两次颜色一致
    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for i,box in enumerate(boxes):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon([box[0][0], box[0][1],
                            box[1][0], box[1][1],
                            box[2][0], box[2][1],
                            box[3][0], box[3][1]], outline=color)
        if txts is not None:
            txt = str(txts[i])
            box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
            box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
            if box_height > 2 * box_width:
                font_size = max(int(box_width * 0.9), 10)
                font = ImageFont.truetype(get_font_file(), font_size, encoding="utf-8")
                cur_y = box[0][1]
                for c in txt:
                    char_size = font.getsize(c)
                    draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                    cur_y += char_size[1]
            else:
                font_size = max(int(box_height * 0.8), 10)
                font = ImageFont.truetype(get_font_file(), font_size, encoding="utf-8")
                draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    if txts is not None:
        if pos == "horizontal":
            img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
            img_show.paste(img_left, (0, 0, w, h))
            img_show.paste(img_right, (w, 0, w * 2, h))
        else:
            img_show = Image.new('RGB', (w, h * 2), (255, 255, 255))
            img_show.paste(img_left, (0, 0, w, h))
            img_show.paste(img_right, (0, h, w , h * 2))
    else:
        img_show = np.array(img_left)
    return np.array(img_show)



def show_img(imgs: np.ndarray, title='img'):
    from matplotlib import pyplot as plt
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path