from torch import nn
import torch


class CTCLoss(nn.Module):

    def __init__(self, blank_idx, reduction='mean'):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction)

    def forward(self, pred, label):
        labels = torch.IntTensor([])
        for j in range(label.size(0)):
            labels = torch.cat((labels, label[j]), 0)
        output_size = torch.IntTensor([pred.size(0)] * int(pred.size(1)))
        label_size = torch.IntTensor([label.size(1)] * int(label.size(0)))
        return self.loss_func(pred, labels, output_size, label_size)
