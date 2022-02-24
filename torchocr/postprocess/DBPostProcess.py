import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from pyclipper import PyclipperOffset
import math
import operator
from functools import reduce


def clockwise_sort_points(_point_coordinates):
    """
        以左上角为起点的顺时针排序
        原理就是将笛卡尔坐标转换为极坐标，然后对极坐标的φ进行排序
    Args:
        _point_coordinates:  待排序的点[(x,y),]
    Returns:    排序完成的点
    """
    center_point = tuple(
        map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), _point_coordinates),
            [len(_point_coordinates)] * 2))
    return sorted(_point_coordinates, key=lambda coord: (180 + math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center_point))[::-1]))) % 360)


class DistillationDBPostProcess(object):
    def __init__(self, model_name=None,
                 key=None,
                 thresh=0.3,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        if model_name is None:
            model_name = ["student"]
        self.model_name = model_name
        self.key = key
        self.post_process = DBPostProcess(thresh=thresh,
                                          box_thresh=box_thresh,
                                          max_candidates=max_candidates,
                                          unclip_ratio=unclip_ratio,
                                          use_dilation=use_dilation,
                                          score_mode=score_mode)

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k].detach().cpu().numpy(), shape_list=shape_list)
        return results


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.6,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        bitmap = (bitmap * 255).astype(np.uint8)
        # structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # bitmap = cv2.morphologyEx(bitmap, cv2.MORPH_CLOSE, structure_element)

        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        elif cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise NotImplementedError(f'opencv {cv2.__version__} not support')

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            # score = self.box_score_fast(pred, points.reshape(-1, 2))
            score = self.box_score_slow(pred, contour)
            if score < self.box_thresh:
                continue
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        try:
            rotated_box = cv2.minAreaRect(contour)
        except:
            print(len(contour))
            return None, 0
        box_points = cv2.boxPoints(rotated_box)
        rotated_points = clockwise_sort_points(box_points)
        rotated_points = list(rotated_points)
        return rotated_points, min(rotated_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict
        pred = pred[:, 0, :, :]
        segmentation = np.zeros_like(pred, dtype=np.float32)
        np.putmask(segmentation, pred > self.thresh, pred)

        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h, )
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch
