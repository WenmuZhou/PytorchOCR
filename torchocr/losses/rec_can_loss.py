import torch
import torch.nn as nn
import numpy as np


class CANLoss(nn.Module):
    '''
    CANLoss is consist of two part:
        word_average_loss: average accuracy of the symbol
        counting_loss: counting loss of every symbol
    '''

    def __init__(self):
        super(CANLoss, self).__init__()

        self.use_label_mask = False
        self.out_channel = 111
        self.cross = nn.CrossEntropyLoss(
            reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')
        self.ratio = 16

    def forward(self, preds, batch):
        word_probs = preds[0]
        counting_preds = preds[1]
        counting_preds1 = preds[2]
        counting_preds2 = preds[3]
        labels = batch[2]
        labels_mask = batch[3]
        counting_labels = gen_counting_label(labels, self.out_channel, True)
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)

        word_loss = self.cross(
            torch.reshape(word_probs, [-1, word_probs.shape[-1]]),
            torch.reshape(labels, [-1]))
        word_average_loss = torch.sum(
            torch.reshape(word_loss * labels_mask, [-1])) / (
                torch.sum(labels_mask) + 1e-10
            ) if self.use_label_mask else word_loss
        loss = word_average_loss + counting_loss
        return {'loss': loss}


def gen_counting_label(labels, channel, tag):
    b, t = labels.shape
    counting_labels = np.zeros([b, channel])

    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    counting_labels = torch.tensor(counting_labels, dtype=torch.float32)
    return counting_labels
