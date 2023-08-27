from torch import nn


class PRENLoss(nn.Module):
    def __init__(self, **kwargs):
        super(PRENLoss, self).__init__()
        # note: 0 is padding idx
        self.loss_func = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def forward(self, predicts, batch):
        loss = self.loss_func(predicts, batch[1].astype('int64'))
        return {'loss': loss}
