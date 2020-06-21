# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 21:16
# @Author  : lgcy
from torch import nn
import torch

def Dice_loss(gt_score, pred_score, gt_mask):
    inter = torch.sum(gt_score * pred_score * gt_mask)
    union = torch.sum(gt_score * gt_mask) + torch.sum(pred_score * gt_mask) + 1e-5
    dice_loss = 1. - (2 * inter / union)
    return dice_loss


def Smooth_l1_loss(gt_geo, pred_geo, gt_score):
    channels = 8
    l_geo_split = torch.split(gt_geo, 1, 1)
    f_geo_split = torch.split(gt_geo, 1, 1)
    smooth_l1 = 0

    for i in range(channels):
        geo_diff = l_geo_split[i] - f_geo_split[i]
        abs_geo_diff = torch.abs(geo_diff)
        smooth_l1_sign = torch.lt(abs_geo_diff, gt_score)
        smooth_l1_sign = smooth_l1_sign.float()
        in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + \
                  (abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
        out_loss = l_geo_split[-1] / channels * in_loss * gt_score
        smooth_l1 += out_loss

    geo_loss = torch.mean(smooth_l1 * gt_score)
    return geo_loss


class EASTLoss(nn.Module):
      '''
      EAST QUAD LOSS
      '''
      def __init__(self, ratio = 0.01):
          super(EASTLoss, self).__init__()
          self.ratio = ratio

      def forward(self, gt_score, pred_score, gt_geo, pred_geo, gt_mask):
          score_loss = Dice_loss(gt_score, pred_score, gt_mask)
          geo_loss = Smooth_l1_loss(gt_geo, pred_geo, gt_score)
          score_loss = score_loss * self.ratio
          total_loss = score_loss + geo_loss
          return total_loss
          
