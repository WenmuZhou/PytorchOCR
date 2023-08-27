import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import L1Loss
from torch.nn import MSELoss as L2Loss
from torch.nn import SmoothL1Loss

class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=137):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()

class KLJSLoss(object):
    def __init__(self, mode='kl'):
        assert mode in ['kl', 'js', 'KL', 'JS'
                        ], "mode can only be one of ['kl', 'KL', 'js', 'JS']"
        self.mode = mode

    def __call__(self, p1, p2, reduction="mean", eps=1e-5):

        if self.mode.lower() == 'kl':
            loss = torch.multiply(p2,torch.log((p2 + eps) / (p1 + eps) + eps))
            loss += torch.multiply(p1, torch.log((p1 + eps) / (p2 + eps) + eps))
            loss *= 0.5
        elif self.mode.lower() == "js":
            loss = torch.multiply(
                p2, torch.log((2 * p2 + eps) / (p1 + p2 + eps) + eps))
            loss += torch.multiply(
                p1, torch.log((2 * p1 + eps) / (p1 + p2 + eps) + eps))
            loss *= 0.5
        else:
            raise ValueError(
                "The mode.lower() if KLJSLoss should be one of ['kl', 'js']")

        if reduction == "mean":
            loss = torch.mean(loss, dim=[1, 2])
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = torch.sum(loss, dim=[1, 2])

        return loss


class DMLLoss(nn.Module):
    """
    DMLLoss
    """

    def __init__(self, act=None, use_log=False):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log
        self.jskl_loss = KLJSLoss(mode="kl")

    def _kldiv(self, x, target):
        eps = 1.0e-10
        loss = target * (torch.log(target + eps) - x)
        # batch mean loss
        loss = torch.sum(loss) / loss.shape[0]
        return loss

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1) + 1e-10
            out2 = self.act(out2) + 1e-10
        if self.use_log:
            # for recognition distillation, log is needed for feature map
            log_out1 = torch.log(out1)
            log_out2 = torch.log(out2)
            loss = (
                self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0
        else:
            # for detection distillation log is not needed
            loss = self.jskl_loss(out1, out2)
        return loss


class DistanceLoss(nn.Module):
    """
    DistanceLoss:
        mode: loss mode
    """

    def __init__(self, mode="l2", **kargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = nn.MSELoss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)


class LossFromOutput(nn.Module):
    def __init__(self, key='loss', reduction='none'):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def forward(self, predicts, batch):
        loss = predicts
        if self.key is not None and isinstance(predicts, dict):
            loss = loss[self.key]
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return {'loss': loss}


class KLDivLoss(nn.Module):
    """
    KLDivLoss
    """

    def __init__(self):
        super().__init__()

    def _kldiv(self, x, target, mask=None):
        eps = 1.0e-10
        loss = target * (torch.log(target + eps) - x)
        if mask is not None:
            loss = loss.flatten(0, 1).sum(dim=1)
            loss = loss.masked_select(mask).mean()
        else:
            # batch mean loss
            loss = torch.sum(loss) / loss.shape[0]
        return loss

    def forward(self, logits_s, logits_t, mask=None):
        log_out_s = F.log_softmax(logits_s, dim=-1)
        out_t = F.softmax(logits_t, dim=-1)
        loss = self._kldiv(log_out_s, out_t, mask)
        return loss


class DKDLoss(nn.Module):
    """
    KLDivLoss
    """

    def __init__(self, temperature=1.0, alpha=1.0, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdim=True)
        t2 = (t * mask2).sum(dim=1, keepdim=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def _kl_div(self, x, label, mask=None):
        y = (label * (torch.log(label + 1e-10) - x)).sum(dim=1)
        if mask is not None:
            y = y.masked_select(mask).mean()
        else:
            y = y.mean()
        return y

    def forward(self, logits_student, logits_teacher, target, mask=None):
        gt_mask = F.one_hot(
            target.reshape([-1]), num_classes=logits_student.shape[-1])
        other_mask = 1 - gt_mask
        logits_student = logits_student.flatten(0, 1)
        logits_teacher = logits_teacher.flatten(0, 1)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = self._kl_div(log_pred_student,
                                 pred_teacher) * (self.temperature**2)
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1)
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1)
        nckd_loss = self._kl_div(log_pred_student_part2,
                                 pred_teacher_part2) * (self.temperature**2)

        loss = self.alpha * tckd_loss + self.beta * nckd_loss

        return loss
