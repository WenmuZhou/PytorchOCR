import torch
from torch import nn


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-12

    def forward(self, x1, x2, target):
        similarity = torch.sum(
            x1 * x2, dim=-1) / (torch.norm(
            x1, dim=-1) * torch.norm(
            x2, dim=-1) + self.epsilon)
        one_list = torch.full_like(target, fill_value=1)
        out = torch.mean(torch.where(torch.equal(target, one_list), 1. - similarity,
                                     torch.maximum(torch.zeros_like(similarity), similarity - self.margin)))
        return out


class AsterLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 ignore_index=-100,
                 sequence_normalize=False,
                 sample_normalize=True,
                 **kwargs):
        super(AsterLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        self.loss_sem = CosineEmbeddingLoss()
        self.is_cosin_loss = True
        self.loss_func_rec = nn.CrossEntropyLoss(weight=None, reduction='none')

    def forward(self, predicts, batch):
        targets = batch[1].astype("int64")
        label_lengths = batch[2].astype('int64')
        sem_target = batch[3].astype('float32')
        embedding_vectors = predicts['embedding_vectors']
        rec_pred = predicts['rec_pred']

        if not self.is_cosin_loss:
            sem_loss = torch.sum(self.loss_sem(embedding_vectors, sem_target))
        else:
            label_target = torch.ones([embedding_vectors.shape[0]])
            sem_loss = torch.sum(
                self.loss_sem(embedding_vectors, sem_target, label_target))

        # rec loss
        batch_size, def_max_length = targets.shape[0], targets.shape[1]

        mask = torch.zeros([batch_size, def_max_length])
        for i in range(batch_size):
            mask[i, :label_lengths[i]] = 1
        mask = mask.float()
        max_length = max(label_lengths)
        assert max_length == rec_pred.shape[1]
        targets = targets[:, :max_length]
        mask = mask[:, :max_length]
        rec_pred = torch.reshape(rec_pred, [-1, rec_pred.shape[2]])
        input = nn.functional.log_softmax(rec_pred, dim=1)
        targets = torch.reshape(targets, [-1, 1])
        mask = torch.reshape(mask, [-1, 1])
        output = -torch.index_select(input, index=targets) * mask
        output = torch.sum(output)
        if self.sequence_normalize:
            output = output / torch.sum(mask)
        if self.sample_normalize:
            output = output / batch_size

        loss = output + sem_loss * 0.1
        return {'loss': loss}
