from torch import nn


class TableMasterLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(TableMasterLoss, self).__init__()
        self.structure_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='mean')
        self.box_loss = nn.L1Loss(reduction='sum')
        self.eps = 1e-12

    def forward(self, predicts, batch):
        # structure_loss
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1]
        structure_targets = structure_targets[:, 1:]
        structure_probs = structure_probs.reshape(
            [-1, structure_probs.shape[-1]])
        structure_targets = structure_targets.reshape([-1])

        structure_loss = self.structure_loss(structure_probs, structure_targets)
        structure_loss = structure_loss.mean()
        losses = dict(structure_loss=structure_loss)

        # box loss
        bboxes_preds = predicts['loc_preds']
        bboxes_targets = batch[2][:, 1:, :]
        bbox_masks = batch[3][:, 1:]
        # mask empty-bbox or non-bbox structure token's bbox.

        masked_bboxes_preds = bboxes_preds * bbox_masks
        masked_bboxes_targets = bboxes_targets * bbox_masks

        # horizon loss (x and width)
        horizon_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 0::2],
                                         masked_bboxes_targets[:, :, 0::2])
        horizon_loss = horizon_sum_loss / (bbox_masks.sum() + self.eps)
        # vertical loss (y and height)
        vertical_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 1::2],
                                          masked_bboxes_targets[:, :, 1::2])
        vertical_loss = vertical_sum_loss / (bbox_masks.sum() + self.eps)

        horizon_loss = horizon_loss.mean()
        vertical_loss = vertical_loss.mean()
        all_loss = structure_loss + horizon_loss + vertical_loss
        losses.update({
            'loss': all_loss,
            'horizon_bbox_loss': horizon_loss,
            'vertical_bbox_loss': vertical_loss
        })
        return losses
