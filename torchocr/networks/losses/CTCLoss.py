from torch import nn
import torch


class CTCLoss(nn.Module):

    def __init__(self, blank_idx, reduction='mean'):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred, args):
        batch_size = pred.size(0)
        label, label_length = args['targets'], args['targets_lengths']
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        preds_lengths = torch.tensor([pred.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        return {'loss': loss}