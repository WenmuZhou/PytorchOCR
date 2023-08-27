import torch
import torch.nn.functional as F
from torch import nn


class DRRGLoss(nn.Module):
    def __init__(self, ohem_ratio=3.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.downsample_ratio = 1.0

    def balance_bce_loss(self, pred, gt, mask):
        """Balanced Binary-CrossEntropy Loss.

        Args:
            pred (Tensor): Shape of :math:`(1, H, W)`.
            gt (Tensor): Shape of :math:`(1, H, W)`.
            mask (Tensor): Shape of :math:`(1, H, W)`.

        Returns:
            Tensor: Balanced bce loss.
        """
        assert pred.shape == gt.shape == mask.shape
        assert torch.all(pred >= 0) and torch.all(pred <= 1)
        assert torch.all(gt >= 0) and torch.all(gt <= 1)
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())

        if positive_count > 0:
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = torch.sum(loss * positive)
            negative_loss = loss * negative
            negative_count = min(
                int(negative.sum()), int(positive_count * self.ohem_ratio))
        else:
            positive_loss = torch.tensor(0.0)
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative
            negative_count = 100
        negative_loss, _ = torch.topk(
            negative_loss.reshape([-1]), negative_count)

        balance_loss = (positive_loss + torch.sum(negative_loss)) / (
            float(positive_count + negative_count) + 1e-5)

        return balance_loss

    def gcn_loss(self, gcn_data):
        """CrossEntropy Loss from gcn module.

        Args:
            gcn_data (tuple(Tensor, Tensor)): The first is the
                prediction with shape :math:`(N, 2)` and the
                second is the gt label with shape :math:`(m, n)`
                where :math:`m * n = N`.

        Returns:
            Tensor: CrossEntropy loss.
        """
        gcn_pred, gt_labels = gcn_data
        gt_labels = gt_labels.reshape([-1])
        loss = F.cross_entropy(gcn_pred, gt_labels)

        return loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        batch_size = len(bitmasks)
        results = []

        kernel = []
        for batch_inx in range(batch_size):
            mask = bitmasks[batch_inx]
            # hxw
            mask_sz = mask.shape
            # left, right, top, bottom
            pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
            mask = F.pad(mask, pad, mode='constant', value=0)
            kernel.append(mask)
        kernel = torch.stack(kernel)
        results.append(kernel)

        return results

    def forward(self, preds, labels):
        """Compute Drrg loss.
        """

        assert isinstance(preds, tuple)
        gt_text_mask, gt_center_region_mask, gt_mask, gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map = labels[
            1:8]

        downsample_ratio = self.downsample_ratio

        pred_maps, gcn_data = preds
        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_top_height_map = pred_maps[:, 4, :, :]
        pred_bot_height_map = pred_maps[:, 5, :, :]
        feature_sz = pred_maps.shape

        # bitmask 2 tensor
        mapping = {
            'gt_text_mask': gt_text_mask.float(),
            'gt_center_region_mask': gt_center_region_mask.float(),
            'gt_mask': gt_mask.float(),
            'gt_top_height_map': gt_top_height_map.float(),
            'gt_bot_height_map': gt_bot_height_map.float(),
            'gt_sin_map': gt_sin_map.float(),
            'gt_cos_map': gt_cos_map.float()
        }
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(downsample_ratio - 1.0) < 1e-2:
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key in ['gt_top_height_map', 'gt_bot_height_map']:
                    gt[key] = [item * downsample_ratio for item in gt[key]]
            gt[key] = [item for item in gt[key]]

        scale = torch.sqrt(1.0 / (pred_sin_map**2 + pred_cos_map**2 + 1e-8))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        loss_text = self.balance_bce_loss(
            F.sigmoid(pred_text_region), gt['gt_text_mask'][0],
            gt['gt_mask'][0])

        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0])
        negative_text_mask = ((1 - gt['gt_text_mask'][0]) * gt['gt_mask'][0])
        loss_center_map = F.binary_cross_entropy(
            F.sigmoid(pred_center_region),
            gt['gt_center_region_mask'][0],
            reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center_positive = torch.sum(loss_center_map *
                                              text_mask) / torch.sum(text_mask)
        else:
            loss_center_positive = torch.tensor(0.0)
        loss_center_negative = torch.sum(
            loss_center_map *
            negative_text_mask) / torch.sum(negative_text_mask)
        loss_center = loss_center_positive + 0.5 * loss_center_negative

        center_mask = (gt['gt_center_region_mask'][0] * gt['gt_mask'][0])
        if int(center_mask.sum()) > 0:
            map_sz = pred_top_height_map.shape
            ones = torch.ones(map_sz, dtype=torch.float32)
            loss_top = F.smooth_l1_loss(
                pred_top_height_map / (gt['gt_top_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            loss_bot = F.smooth_l1_loss(
                pred_bot_height_map / (gt['gt_bot_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            gt_height = (
                gt['gt_top_height_map'][0] + gt['gt_bot_height_map'][0])
            loss_height = torch.sum(
                (torch.log(gt_height + 1) *
                 (loss_top + loss_bot)) * center_mask) / torch.sum(center_mask)

            loss_sin = torch.sum(
                F.smooth_l1_loss(
                    pred_sin_map, gt['gt_sin_map'][0],
                    reduction='none') * center_mask) / torch.sum(center_mask)
            loss_cos = torch.sum(
                F.smooth_l1_loss(
                    pred_cos_map, gt['gt_cos_map'][0],
                    reduction='none') * center_mask) / torch.sum(center_mask)
        else:
            loss_height = torch.tensor(0.0)
            loss_sin = torch.tensor(0.0)
            loss_cos = torch.tensor(0.0)

        loss_gcn = self.gcn_loss(gcn_data)

        loss = loss_text + loss_center + loss_height + loss_sin + loss_cos + loss_gcn
        results = dict(
            loss=loss,
            loss_text=loss_text,
            loss_center=loss_center,
            loss_height=loss_height,
            loss_sin=loss_sin,
            loss_cos=loss_cos,
            loss_gcn=loss_gcn)

        return results
