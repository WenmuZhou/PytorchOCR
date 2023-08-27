from torch import nn


class ClsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predicts, batch):
        label = batch[1].astype("int64")
        loss = self.loss_func(input=predicts, label=label)
        return {'loss': loss}
