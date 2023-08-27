import torch
from torch import nn


class SATRNLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SATRNLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)  # 6626
        self.loss_func = nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index)

    def forward(self, predicts, batch):
        predict = predicts[:, :
                           -1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = batch[1].astype(
            "int64")[:, 1:]  # ignore first index of target in loss calculation
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[
            1], predict.shape[2]
        assert len(label.shape) == len(list(predict.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = torch.reshape(predict, [-1, num_classes])
        targets = torch.reshape(label, [-1])
        loss = self.loss_func(inputs, targets)
        return {'loss': loss.mean()}
