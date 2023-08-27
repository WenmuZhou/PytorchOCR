"""
This code is refer from:
https://github.com/shengtao96/CentripetalText/tree/main/models/loss
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def ohem_single(score, gt_text, training_mask):
    # online hard example mining

    pos_num = int(torch.sum(gt_text > 0.5)) - int(
        torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape((1, selected_mask.shape[0], selected_mask.shape[1])).float()
        return selected_mask

    neg_num = int(torch.sum((gt_text <= 0.5) & (training_mask > 0.5)))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape((1, selected_mask.shape[0], selected_mask.shape[1])).float()
        return selected_mask

    # hard example
    neg_score = score[(gt_text <= 0.5) & (training_mask > 0.5)]
    neg_score_sorted = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) |
                     (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape((1, selected_mask.shape[0], selected_mask.shape[1])).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(
            ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[
                i, :, :]))

    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks


def iou_single(a, b, mask, n_class):
    EPS = 1e-6
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []

    # iou of each class
    for i in range(n_class):
        inter = ((a == i) & (b == i)).float()
        union = ((a == i) | (b == i)).float()

        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape((batch_size, -1))
    b = b.reshape((batch_size, -1))
    mask = mask.reshape((batch_size, -1))

    iou = torch.zeros(batch_size).float()
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = torch.mean(iou)
    return iou


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]
        input = F.sigmoid(input)  # scale to 0-1

        input = input.reshape((batch_size, -1))
        target = target.reshape((batch_size, -1)).float()
        mask = mask.reshape((batch_size, -1)).float()

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

        np_coord = np.zeros(shape=[640, 640, 2], dtype=np.int64)
        for i in range(640):
            for j in range(640):
                np_coord[i, j, 0] = j
                np_coord[i, j, 1] = i
        np_coord = np_coord.reshape((-1, 2))

        self.coord = nn.Parameter(torch.from_numpy(np_coord))
        self.coord.requires_grade = False

    def forward_single(self, input, target, mask, beta=1.0, eps=1e-6):
        batch_size = input.shape[0]

        diff = torch.abs(input - target) * mask.unsqueeze(1)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                            diff - 0.5 * beta)
        loss = loss.reshape((batch_size, -1)).float()
        mask = mask.reshape((batch_size, -1)).float()
        loss = torch.sum(loss, dim=-1)
        loss = loss / (mask.sum(dim=-1) + eps)

        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance,
                      training_mask):

        with torch.no_grad():
            select_distance_list = []
            for i in range(2):
                tmp1 = distance[i, :]
                tmp2 = tmp1[self.coord[:, 1], self.coord[:, 0]]
                select_distance_list.append(tmp2.unsqueeze(0))
            select_distance = torch.cat(select_distance_list, dim=0)
            off_points = self.coord.float() + 10 * select_distance.permute((1, 0))

            off_points = off_points.float()
            off_points = torch.clip(off_points, 0, distance.shape[-1] - 1)

            selected_mask = (
                gt_instance[self.coord[:, 1], self.coord[:, 0]] !=
                gt_kernel_instance[off_points[:, 1], off_points[:, 0]])
            selected_mask = selected_mask.reshape((1, -1, distance.shape[-1])).long()
            selected_training_mask = selected_mask * training_mask

            return selected_training_mask

    def forward(self,
                distances,
                gt_instances,
                gt_kernel_instances,
                training_masks,
                gt_distances,
                reduce=True):

        selected_training_masks = []
        for i in range(distances.shape[0]):
            selected_training_masks.append(
                self.select_single(distances[i, :, :, :], gt_instances[i, :, :],
                                   gt_kernel_instances[i, :, :], training_masks[
                                       i, :, :]))
        selected_training_masks = torch.cat(selected_training_masks, 0).float()

        loss = self.forward_single(distances, gt_distances,
                                   selected_training_masks, self.beta)
        loss = self.loss_weight * loss

        with torch.no_grad():
            batch_size = distances.shape[0]
            false_num = selected_training_masks.reshape((batch_size, -1))
            false_num = false_num.sum(dim=-1)
            total_num = training_masks.reshape((batch_size, -1)).float()
            total_num = total_num.sum(dim=-1)
            iou_text = (total_num - false_num) / (total_num + 1e-6)

        if reduce:
            loss = torch.mean(loss)

        return loss, iou_text


class CTLoss(nn.Module):
    def __init__(self):
        super(CTLoss, self).__init__()
        self.kernel_loss = DiceLoss()
        self.loc_loss = SmoothL1Loss(beta=0.1, loss_weight=0.05)

    def forward(self, preds, batch):
        imgs = batch[0]
        out = preds['maps']
        gt_kernels, training_masks, gt_instances, gt_kernel_instances, training_mask_distances, gt_distances = batch[
            1:]

        kernels = out[:, 0, :, :]
        distances = out[:, 1:, :, :]

        # kernel loss
        selected_masks = ohem_batch(kernels, gt_kernels, training_masks)

        loss_kernel = self.kernel_loss(
            kernels, gt_kernels, selected_masks, reduce=False)

        iou_kernel = iou((kernels > 0).long(),
                         gt_kernels,
                         training_masks,
                         reduce=False)
        losses = dict(loss_kernels=loss_kernel, )

        # loc loss
        loss_loc, iou_text = self.loc_loss(
            distances,
            gt_instances,
            gt_kernel_instances,
            training_mask_distances,
            gt_distances,
            reduce=False)
        losses.update(dict(loss_loc=loss_loc, ))

        loss_all = loss_kernel + loss_loc
        losses = {'loss': loss_all}

        return losses
