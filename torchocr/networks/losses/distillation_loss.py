import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .DBLoss import DBLoss

def _sum_loss(loss_dict):
    if "loss" in loss_dict.keys():
        return loss_dict
    else:
        loss_dict["loss"] = 0.
        for k, value in loss_dict.items():
            if k == "loss":
                continue
            else:
                loss_dict["loss"] += value
        return loss_dict


class KLJSLoss(object):
    def __init__(self, mode='kl'):
        assert mode in ['kl', 'js', 'KL', 'JS'
                        ], "mode can only be one of ['kl', 'js', 'KL', 'JS']"
        self.mode = mode

    def __call__(self, p1, p2, reduction="mean"):

        loss = torch.mul(p2, torch.log((p2 + 1e-5) / (p1 + 1e-5) + 1e-5))

        if self.mode.lower() == "js":
            loss += torch.mul(
                p1, torch.log((p1 + 1e-5) / (p2 + 1e-5) + 1e-5))
            loss *= 0.5
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = torch.sum(loss)

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
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log

        self.jskl_loss = KLJSLoss(mode="js")

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1)
            out2 = self.act(out2)
        if self.use_log:
            # for recognition distillation, log is needed for feature map
            log_out1 = torch.log(out1)
            log_out2 = torch.log(out2)
            loss = (F.kl_div(
                log_out1, out2, reduction='batchmean') + F.kl_div(
                log_out2, out1, reduction='batchmean')) / 2.0
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


class DistillationDMLLoss(DMLLoss):
    """
    """

    def __init__(self,
                 model_name_pairs=[],
                 act=None,
                 use_log=False,
                 key=None,
                 maps_name=None,
                 name="dml"):
        super().__init__(act=act, use_log=use_log)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif type(maps_name) == str:
            return [maps_name]
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.maps_name is None:
                loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1],idx)] = loss[key]
                else:
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict["{}_{}_{}_{}_{}".format(key, pair[0], pair[1], self.maps_name[_c], idx)] = loss[key]
                    else:
                        loss_dict["{}_{}_{}".format(self.name, self.maps_name[_c], idx)] = loss

        loss_dict = _sum_loss(loss_dict)

        return loss_dict


class DistillationDBLoss(DBLoss):
    def __init__(self,
                 model_name_list=[],
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 name="db",
                 **kwargs):
        super().__init__()
        self.model_name_list = model_name_list
        self.name = name
        self.key = None

    def forward(self, predicts, batch):
        loss_dict = {}
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss.keys():
                    if key == "loss":
                        continue
                    name = "{}_{}_{}".format(self.name, model_name, key)
                    loss_dict[name] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss

        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationDilaDBLoss(DBLoss):
    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 name="dila_dbloss"):
        super().__init__()
        self.model_name_pairs = model_name_pairs
        self.name = name
        self.key = key

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            # stu_outs = predicts[pair[0]]
            # tch_outs = predicts[pair[1]]
            # if self.key is not None:
            #     stu_preds = stu_outs[self.key]
            #     tch_preds = tch_outs[self.key]
            stu_preds = predicts[pair[0]]
            tch_preds = predicts[pair[1]]
            stu_shrink_maps = stu_preds[:, 0, :, :]
            stu_binary_maps = stu_preds[:, 2, :, :]

            # dilation to teacher prediction
            dilation_w = np.array([[1, 1], [1, 1]])
            th_shrink_maps = tch_preds[:, 0, :, :]
            th_shrink_maps = th_shrink_maps.cpu().detach().numpy() > 0.3  # thresh = 0.3
            dilate_maps = np.zeros_like(th_shrink_maps).astype(np.float32)
            for i in range(th_shrink_maps.shape[0]):
                dilate_maps[i] = cv2.dilate(
                    th_shrink_maps[i, :, :].astype(np.uint8), dilation_w)
            th_shrink_maps = torch.tensor(dilate_maps).cuda()

            label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = batch['threshold_map'], batch['threshold_mask'], batch['shrink_map'], batch['shrink_mask']

            # calculate the shrink map loss
            bce_loss = self.alpha * self.bce_loss(
                stu_shrink_maps, th_shrink_maps, label_shrink_mask)
            loss_binary_maps = self.dice_loss(stu_binary_maps, th_shrink_maps,
                                              label_shrink_mask)

            # k = f"{self.name}_{pair[0]}_{pair[1]}"
            k = "{}_{}_{}".format(self.name, pair[0], pair[1])
            loss_dict[k] = bce_loss + loss_binary_maps

        loss_dict = _sum_loss(loss_dict)
        return loss_dict


class DistillationDistanceLoss(DistanceLoss):
    """
    """

    def __init__(self,
                 mode="l2",
                 model_name_pairs=[],
                 key=None,
                 name="loss_distance",
                 **kargs):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + "_l2"

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            # if self.key is not None:
            #     out1 = out1[self.key]
            #     out2 = out2[self.key]
            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, key, idx)] = loss[
                        key]
            else:
                loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1],
                                               idx)] = loss
        return loss_dict
