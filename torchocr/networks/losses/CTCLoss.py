from torch import nn
import torch


class CTCLoss(nn.Module):

    def __init__(self, blank_idx, reduction='mean'):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred, args):
        label, label_length = args[0], args[1]
        pred = pred.transpose(0,1)
        batch_size = label.size(0)
        pred_size = torch.LongTensor([pred.size(0)] * batch_size)

        return self.loss_func(pred, label, pred_size, label_length) / batch_size
