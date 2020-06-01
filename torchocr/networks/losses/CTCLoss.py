from torch import nn
import torch


class CTCLoss(nn.Module):

    def __init__(self, blank_idx, reduction='mean'):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction)

    def forward(self, pred, args):
        label, label_length = args[0], args[1]

        pred = pred.transpose(1, 0).log_softmax(2).detach().requires_grad_()

        labels = torch.IntTensor([])
        label_size = []
        for j in range(label.size(0)):
            labels = torch.cat((labels, label[j][:label_length[j]]), 0)
            label_size.append(label_length[j])

        label_size = torch.IntTensor(label_size)
        output_size = torch.IntTensor([pred.size(0)] * int(pred.size(1)))

        return self.loss_func(pred, labels, output_size, label_size)
