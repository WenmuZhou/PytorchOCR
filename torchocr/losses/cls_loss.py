from torch import nn


class ClsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predicts, batch):
        label = batch[1].long()
        loss = self.loss_func(predicts['res'], label)
        return {'loss': loss}
