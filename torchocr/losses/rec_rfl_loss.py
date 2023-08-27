import torch
from torch import nn

from .basic_loss import CELoss, DistanceLoss


class RFLLoss(nn.Module):
    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__()

        self.cnt_loss = nn.MSELoss(**kwargs)
        self.seq_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predicts, batch):

        self.total_loss = {}
        total_loss = 0.0
        if isinstance(predicts, tuple) or isinstance(predicts, list):
            cnt_outputs, seq_outputs = predicts
        else:
            cnt_outputs, seq_outputs = predicts, None
        # batch [image, label, length, cnt_label]
        if cnt_outputs is not None:
            cnt_loss = self.cnt_loss(cnt_outputs,batch[3].float())
            self.total_loss['cnt_loss'] = cnt_loss
            total_loss += cnt_loss

        if seq_outputs is not None:
            targets = batch[1].astype("int64")
            label_lengths = batch[2].astype('int64')
            batch_size, num_steps, num_classes = seq_outputs.shape[
                0], seq_outputs.shape[1], seq_outputs.shape[2]
            assert len(targets.shape) == len(list(seq_outputs.shape)) - 1, \
                "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

            inputs = seq_outputs[:, :-1, :]
            targets = targets[:, 1:]

            inputs = torch.reshape(inputs, [-1, inputs.shape[-1]])
            targets = torch.reshape(targets, [-1])
            seq_loss = self.seq_loss(inputs, targets)
            self.total_loss['seq_loss'] = seq_loss
            total_loss += seq_loss

        self.total_loss['loss'] = total_loss
        return self.total_loss
