import torch
from torch import nn
from torch.nn import functional as F


class TableAttentionLoss(nn.Module):
    def __init__(self, structure_weight, loc_weight, **kwargs):
        super(TableAttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1].astype("int64")
        structure_targets = structure_targets[:, 1:]
        structure_probs = torch.reshape(structure_probs,
                                         [-1, structure_probs.shape[-1]])
        structure_targets = torch.reshape(structure_targets, [-1])
        structure_loss = self.loss_func(structure_probs, structure_targets)

        structure_loss = torch.mean(structure_loss) * self.structure_weight

        loc_preds = predicts['loc_preds']
        loc_targets = batch[2].astype("float32")
        loc_targets_mask = batch[3].astype("float32")
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        loc_loss = F.mse_loss(loc_preds * loc_targets_mask,
                              loc_targets) * self.loc_weight

        total_loss = structure_loss + loc_loss
        return {
            'loss': total_loss,
            "structure_loss": structure_loss,
            "loc_loss": loc_loss
        }


class SLALoss(nn.Module):
    def __init__(self, structure_weight, loc_weight, loc_loss='mse', **kwargs):
        super(SLALoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='mean')
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        self.loc_loss = loc_loss
        self.eps = 1e-12

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1].astype("int64")
        structure_targets = structure_targets[:, 1:]

        structure_loss = self.loss_func(structure_probs, structure_targets)

        structure_loss = torch.mean(structure_loss) * self.structure_weight

        loc_preds = predicts['loc_preds']
        loc_targets = batch[2].astype("float32")
        loc_targets_mask = batch[3].astype("float32")
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]

        loc_loss = F.smooth_l1_loss(
            loc_preds * loc_targets_mask,
            loc_targets * loc_targets_mask,
            reduction='sum') * self.loc_weight

        loc_loss = loc_loss / (loc_targets_mask.sum() + self.eps)
        total_loss = structure_loss + loc_loss
        return {
            'loss': total_loss,
            "structure_loss": structure_loss,
            "loc_loss": loc_loss
        }
