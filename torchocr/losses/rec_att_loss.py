import torch
from torch import nn


class AttentionLoss(nn.Module):
    def __init__(self, **kwargs):
        super(AttentionLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')

    def forward(self, predicts, batch):
        predicts = predicts['res'][:, :-1]
        targets = batch[1].long()[:,1:]
        assert len(targets.shape) == len(list(predicts.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = torch.reshape(predicts, [-1, predicts.shape[-1]])
        targets = torch.reshape(targets, [-1])

        return {'loss': torch.sum(self.loss_func(inputs, targets))}
