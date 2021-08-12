# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun

import os
import math
import random
import numbers
import pathlib
import pyclipper
from torch.utils import data
import glob
import numpy as np
import cv2
from skimage.util import random_noise
import json
from tqdm import tqdm
from torchvision import transforms


# from utils.utils import draw_bbox

# 图像均为cv2读取
class DataAugment():
    def __init__(self):
        pass

    def add_noise(self, im: np.ndarray):
        """
        对图片加噪声
        :param img: 图像array
        :return: 加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        """
        return (random_noise(im, mode='gaussian', clip=True) * 255).astype(im.dtype)

    def random_scale(self, im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray or list) -> tuple:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param im: 原图
        :param text_polys: 文本框
        :param scales: 尺度
        :return: 经过缩放的图片和文本
        """
        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale
        return im, tmp_text_polys

    def random_rotate_img_bbox(self, img, text_polys, degrees: numbers.Number or list or tuple or np.ndarray,
                               same_size=False):
        """
        从给定的角度中选择一个角度，对图片和文本框进行旋转
        :param img: 图片
        :param text_polys: 文本框
        :param degrees: 角度，可以是一个数值或者list
        :param same_size: 是否保持和原图一样大
        :return: 旋转后的图片和角度
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        angle = np.random.uniform(degrees[0], degrees[1])

        if same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_text_polys = list()
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        return rot_img, np.array(rot_text_polys, dtype=np.float32)

    def random_crop_img_bboxes(self, im: np.ndarray, text_polys: np.ndarray, max_tries=50) -> tuple:
        """
        从图片中裁剪出 cropsize大小的图片和对应区域的文本框
        :param im: 图片
        :param text_polys: 文本框
        :param max_tries: 最大尝试次数
        :return: 裁剪后的图片和文本框
        """
        h, w, _ = im.shape
        pad_h = h // 10
        pad_w = w // 10
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
        for poly in text_polys:
            poly = np.round(poly, decimals=0).astype(np.int32)  # 四舍五入取整
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1  # 将文本区域的在w_array上设为1，表示x轴方向上这部分位置有文本
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1  # 将文本区域的在h_array上设为1，表示y轴方向上这部分位置有文本
        # 在两个轴上 拿出背景位置去进行随机的位置选择，避免选择的区域穿过文本
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            # 整张图全是文本的情况下，直接返回
            return im, text_polys
        for i in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            # 对选择区域进行边界控制
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if xmax - xmin < 0.1 * w or ymax - ymin < 0.1 * h:
                # 选择的区域过小
                # area too small
                continue
            if text_polys.shape[0] != 0:  # 这个判断不知道干啥的
                poly_axis_in_area = (text_polys[:, :, 0] >= xmin) & (text_polys[:, :, 0] <= xmax) \
                                    & (text_polys[:, :, 1] >= ymin) & (text_polys[:, :, 1] <= ymax)
                selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []
            if len(selected_polys) == 0:
                # 区域内没有文本
                continue
            im = im[ymin:ymax + 1, xmin:xmax + 1, :]
            polys = text_polys[selected_polys]
            # 坐标调整到裁剪图片上
            polys[:, :, 0] -= xmin
            polys[:, :, 1] -= ymin
            return im, polys
        return im, text_polys

    def random_crop_image_pse(self, im: np.ndarray, text_polys: np.ndarray, input_size) -> tuple:
        """
        从图片中裁剪出 cropsize大小的图片和对应区域的文本框
        :param im: 图片
        :param text_polys: 文本框
        :param input_size: 输出图像大小
        :return: 裁剪后的图片和文本框
        """
        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < input_size:
            # 保证短边 >= inputsize
            scale = input_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            text_polys *= scale
            h, w, _ = im.shape
        # 计算随机范围
        w_range = w - input_size
        h_range = h - input_size
        for _ in range(50):
            xmin = random.randint(0, w_range)
            ymin = random.randint(0, h_range)
            xmax = xmin + input_size
            ymax = ymin + input_size
            if text_polys.shape[0] != 0:
                selected_polys = []
                for poly in text_polys:
                    if poly[:, 0].max() < xmin or poly[:, 0].min() > xmax or \
                            poly[:, 1].max() < ymin or poly[:, 1].min() > ymax:
                        continue
                    # area_p = cv2.contourArea(poly)
                    poly[:, 0] -= xmin
                    poly[:, 1] -= ymin
                    poly[:, 0] = np.clip(poly[:, 0], 0, input_size)
                    poly[:, 1] = np.clip(poly[:, 1], 0, input_size)
                    # rect = cv2.minAreaRect(poly)
                    # area_n = cv2.contourArea(poly)
                    # h1, w1 = rect[1]
                    # if w1 < 10 or h1 < 10 or area_n / area_p < 0.5:
                    #     continue
                    selected_polys.append(poly)
            else:
                selected_polys = []
            # if len(selected_polys) == 0:
            # 区域内没有文本
            # continue
            im = im[ymin:ymax, xmin:xmax, :]
            polys = np.array(selected_polys)
            return im, polys
        return im, text_polys

    def random_crop_author(self, imgs, img_size):
        h, w = imgs[0].shape[0:2]
        th, tw = img_size
        if w == tw and h == th:
            return imgs

        # label中存在文本实例，并且按照概率进行裁剪
        if np.max(imgs[1][:, :, -1]) > 0 and random.random() > 3.0 / 8.0:
            # 文本实例的top left点
            tl = np.min(np.where(imgs[1][:, :, -1] > 0), axis=1) - img_size
            tl[tl < 0] = 0
            # 文本实例的 bottom right 点
            br = np.max(np.where(imgs[1][:, :, -1] > 0), axis=1) - img_size
            br[br < 0] = 0
            # 保证选到右下角点是，有足够的距离进行crop
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)
            for _ in range(50000):
                i = random.randint(tl[0], br[0])
                j = random.randint(tl[1], br[1])
                # 保证最小的图有文本
                if imgs[1][:, :, 0][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # return i, j, th, tw
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        return imgs

    def resize(self, im: np.ndarray, text_polys: np.ndarray,
               input_size: numbers.Number or list or tuple or np.ndarray, keep_ratio: bool = False) -> tuple:
        """
        对图片和文本框进行resize
        :param im: 图片
        :param text_polys: 文本框
        :param input_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param keep_ratio: 是否保持长宽比
        :return: resize后的图片和文本框
        """
        if isinstance(input_size, numbers.Number):
            if input_size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            input_size = (input_size, input_size)
        elif isinstance(input_size, list) or isinstance(input_size, tuple) or isinstance(input_size, np.ndarray):
            if len(input_size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            input_size = (input_size[0], input_size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        if keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, input_size[0])
            max_w = max(w, input_size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, input_size)
        w_scale = input_size[0] / float(w)
        h_scale = input_size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale
        return im, text_polys

    def horizontal_flip(self, im: np.ndarray, text_polys: np.ndarray) -> tuple:
        """
        对图片和文本框进行水平翻转
        :param im: 图片
        :param text_polys: 文本框
        :return: 水平翻转之后的图片和文本框
        """
        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]
        return flip_im, flip_text_polys

    def vertical_flip(self, im: np.ndarray, text_polys: np.ndarray) -> tuple:
        """
         对图片和文本框进行竖直翻转
        :param im: 图片
        :param text_polys: 文本框
        :return: 竖直翻转之后的图片和文本框
        """
        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        return flip_im, flip_text_polys

    def test(self, im: np.ndarray, text_polys: np.ndarray):
        print('随机尺度缩放')
        t_im, t_text_polys = self.random_scale(im, text_polys, [0.5, 1, 2, 3])
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_scale')

        print('随机旋转')
        t_im, t_text_polys = self.random_rotate_img_bbox(im, text_polys, 10)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_rotate_img_bbox')

        print('随机裁剪')
        t_im, t_text_polys = self.random_crop_img_bboxes(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_crop_img_bboxes')

        print('水平翻转')
        t_im, t_text_polys = self.horizontal_flip(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'horizontal_flip')

        print('竖直翻转')
        t_im, t_text_polys = self.vertical_flip(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'vertical_flip')
        show_pic(im, text_polys, 'vertical_flip_ori')

        print('加噪声')
        t_im = self.add_noise(im)
        print(t_im.shape)
        show_pic(t_im, text_polys, 'add_noise')
        show_pic(im, text_polys, 'add_noise_ori')


data_aug = DataAugment()


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)


def generate_rbox(im_size, text_polys, text_tags, training_mask, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是北京
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    for poly, tag in zip(text_polys, text_tags):
        poly = poly.astype(np.int)
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))
        cv2.fillPoly(score_map, shrinked_poly, 1)
        # 制作mask
        # rect = cv2.minAreaRect(shrinked_poly)
        # poly_h, poly_w = rect[1]

        # if min(poly_h, poly_w) < 10:
        #     cv2.fillPoly(training_mask, shrinked_poly, 0)
        if tag:
            cv2.fillPoly(training_mask, shrinked_poly, 0)
        # 闭运算填充内部小框
        # kernel = np.ones((3, 3), np.uint8)
        # score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel)
    return score_map, training_mask


def augmentation(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray, degrees: int, input_size: int) -> tuple:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    # 640 × 640 random samples are cropped from the transformed images
    # im, text_polys = data_aug.random_crop_img_bboxes(im, text_polys)

    # im, text_polys = data_aug.resize(im, text_polys, input_size, keep_ratio=False)
    # im, text_polys = data_aug.random_crop_image_pse(im, text_polys, input_size)

    return im, text_polys
class EastRandomCropData():
    def __init__(self, size=(640, 640), max_tries=50, min_crop_side_ratio=0.1, require_original_image=False, keep_ratio=True):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image
        self.keep_ratio = keep_ratio

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        training_mask = data['training_mask']
        score_maps = data['score_maps'].transpose((1,2,0))
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']
        all_care_polys = [text_polys[i] for i, tag in enumerate(ignore_tags) if not tag]
        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        try:
            if self.keep_ratio:
                if len(im.shape) == 3:
                    padimg = np.zeros((self.size[1], self.size[0], im.shape[2]), im.dtype)
                else:
                    padimg = np.zeros((self.size[1], self.size[0]), im.dtype)
                padimg[:h, :w] = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
                img = padimg

                padimg2 = np.zeros((self.size[1], self.size[0]), im.dtype)
                padimg2[:h, :w] = cv2.resize(training_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
                data['training_mask'] = padimg2

                padimg2 = np.zeros((self.size[1], self.size[0],6), im.dtype)
                padimg2[:h, :w] = cv2.resize(score_maps[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
                data['score_maps'] = padimg2.transpose((2,0,1))
            else:
                img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], tuple(self.size))
        except Exception:
            import traceback
            traceback.print_exc()
        # crop 文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        try:
            for poly, text, tag in zip(text_polys, texts, ignore_tags):
                poly = ((np.array(poly) - (crop_x, crop_y)) * scale).astype('float32')
                if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                    text_polys_crop.append(poly)
                    ignore_tags_crop.append(tag)
                    texts_crop.append(text)
            data['img'] = img
            data['text_polys'] = text_polys_crop
            data['ignore_tags'] = ignore_tags_crop
            data['texts'] = texts_crop
        except:
            a = 1
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys):
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

erc=EastRandomCropData()
def image_label(data, n: int, m: float, input_size: int,
                defrees: int = 10,
                scales: np.ndarray = np.array([0.5, 1, 2.0, 3.0])) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''


    im = cv2.imread(data['img_path'])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # 检查越界
    data['text_polys'] = check_and_validate_polys(data['text_polys'], (h, w))
    data['img'], data['text_polys'], = augmentation(im, data['text_polys'], scales, defrees, input_size)

    h, w, _ = data['img'].shape
    short_edge = min(h, w)
    if isinstance(input_size, dict):
        print(input_size)
        pass
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        data['img'] = cv2.resize(data['img'], dsize=None, fx=scale, fy=scale)
        data['text_polys'] *= scale
    h, w, _ = data['img'].shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask = generate_rbox((h, w), data['text_polys'], data['ignore_tags'], training_mask, i, n, m)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)
    data['training_mask']=training_mask
    data['score_maps']=score_maps
    data=erc(data)
    return data


    # imgs = data_aug.random_crop_author([im, score_maps.transpose((1, 2, 0)), training_mask], (input_size, input_size))
    # return imgs[0], imgs[1].transpose((2, 0, 1)), imgs[2], text_polys, text_tags  # im,score_maps,training_mask#

import torch
class MyDataset(data.Dataset):
    def __init__(self, config):
        self.load_char_annotation = False
        self.data_list = self.load_data(config.file)
        self.data_shape = config.data_shape
        self.filter_keys = config.filter_keys
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])
        self.n = config.n
        self.m = config.m

    def __getitem__(self, index):
        # print(self.image_list[index])
        data = self.data_list[index]
        img_path, text_polys, text_tags = self.data_list[index]['img_path'], self.data_list[index]['text_polys'], self.data_list[index]['ignore_tags']
        data = image_label(data, input_size=self.data_shape,n=self.n,m=self.m)

        im = cv2.imread(img_path)
        if self.transform:
            img = self.transform(data['img'])
        shape = (data['img'].shape[0], data['img'].shape[1])

        data['img'] = img
        data['shape'] = shape
        # data['score_maps'] = score_maps
        # data['training_mask'] = training_mask
        # data['text_polys'] =torch.Tensor(list(text_polys))
        # data['ignore_tags'] = [text_tags]
        # data['shape'] = shape
        # data['texts'] = [data['texts']]

        if len(self.filter_keys):
            data_dict = {}
            for k, v in data.items():
                if k not in self.filter_keys:
                    data_dict[k] = v
            return data_dict
        else:
            # return {'img': img, 'score_maps': score_maps, 'training_mask': training_mask, 'shape': shape, 'text_polys': list(text_polys), 'ignore_tags': text_tags}
            return {}

    def load_data(self, path: str) -> list:
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
            data_list.append({'img_path': img_path, 'img_name': gt['img_name'], 'text_polys': np.array(polygons, dtype=np.float32),
                              'texts': texts, 'ignore_tags': illegibility_list})
        return data_list

    def __len__(self):
        return len(self.data_list)

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img


def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


if __name__ == '__main__':
    import torch
    import config
    from config.cfg_det_pse import config
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms

    train_data = MyDataset(config.dataset.train.dataset)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i, batch_data in enumerate(train_loader):
        img, label, mask = batch_data['img'], batch_data['score_maps'], batch_data['training_mask']
        print(label.shape)
        print(img.shape)
        print(label[0][-1].sum())
        print(mask[0].shape)
        pbar.update(1)
        show_img((img[0] * mask[0].to(torch.float)).numpy().transpose(1, 2, 0), color=True)
        show_img(label[0])
        show_img(mask[0])
        plt.show()

    pbar.close()
