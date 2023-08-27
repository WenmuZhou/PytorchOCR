import torch
from torch import nn
from .det_basic_loss import DiceLoss


class EASTLoss(nn.Module):
    """
    """

    def __init__(self,
                 eps=1e-6,
                 **kwargs):
        super(EASTLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, predicts, labels):
        l_score, l_geo, l_mask = labels[1:]
        f_score = predicts['f_score']
        f_geo = predicts['f_geo']

        dice_loss = self.dice_loss(f_score, l_score, l_mask)

        #smoooth_l1_loss
        channels = 8
        l_geo_split = torch.split(l_geo, split_size_or_sections=channels + 1, dim=1)
        f_geo_split = torch.split(f_geo, split_size_or_sections=channels, dim=1)
        smooth_l1 = 0
        for i in range(0, channels):
            geo_diff = l_geo_split[i] - f_geo_split[i]
            abs_geo_diff = torch.abs(geo_diff)
            smooth_l1_sign = (abs_geo_diff < l_score).float()
            in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + \
                (abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
            out_loss = l_geo_split[-1] / channels * in_loss * l_score
            smooth_l1 += out_loss
        smooth_l1_loss = torch.mean(smooth_l1 * l_score)

        dice_loss = dice_loss * 0.01
        total_loss = dice_loss + smooth_l1_loss
        losses = {"loss":total_loss, \
                  "dice_loss":dice_loss,\
                  "smooth_l1_loss":smooth_l1_loss}
        return losses
