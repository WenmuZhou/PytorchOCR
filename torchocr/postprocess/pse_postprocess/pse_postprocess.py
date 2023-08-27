import numpy as np
import cv2
import torch
from torch.nn import functional as F

from torchocr.postprocess.pse_postprocess.pse import pse


class PSEPostProcess(object):
    """
    The post process for PSE.
    """

    def __init__(self,
                 thresh=0.5,
                 box_thresh=0.85,
                 min_area=16,
                 box_type='quad',
                 scale=4,
                 **kwargs):
        assert box_type in ['quad', 'poly'], 'Only quad and poly is supported'
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.min_area = min_area
        self.box_type = box_type
        self.scale = scale

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        pred = F.interpolate(
            pred, scale_factor=4 // self.scale, mode='bilinear')

        score = F.sigmoid(pred[:, 0, :, :])

        kernels = (pred > self.thresh).astype('float32')
        text_mask = kernels[:, 0, :, :]
        text_mask = torch.unsqueeze(text_mask, dim=1)

        kernels[:, 0:, :, :] = kernels[:, 0:, :, :] * text_mask

        score = score.numpy()
        kernels = kernels.numpy().astype(np.uint8)

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            boxes, scores = self.boxes_from_bitmap(score[batch_index],
                                                   kernels[batch_index],
                                                   shape_list[batch_index])

            boxes_batch.append({'points': boxes, 'scores': scores})
        return boxes_batch

    def boxes_from_bitmap(self, score, kernels, shape):
        label = pse(kernels, self.min_area)
        return self.generate_box(score, label, shape)

    def generate_box(self, score, label, shape):
        src_h, src_w, ratio_h, ratio_w = shape
        label_num = np.max(label) + 1

        boxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < self.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < self.box_thresh:
                label[ind] = 0
                continue

            if self.box_type == 'quad':
                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect)
            elif self.box_type == 'poly':
                box_height = np.max(points[:, 1]) + 10
                box_width = np.max(points[:, 0]) + 10

                mask = np.zeros((box_height, box_width), np.uint8)
                mask[points[:, 1], points[:, 0]] = 255

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                bbox = np.squeeze(contours[0], 1)
            else:
                raise NotImplementedError

            bbox[:, 0] = np.clip(np.round(bbox[:, 0] / ratio_w), 0, src_w)
            bbox[:, 1] = np.clip(np.round(bbox[:, 1] / ratio_h), 0, src_h)
            boxes.append(bbox)
            scores.append(score_i)
        return boxes, scores
