import torch
import torch.nn as nn


class ACELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(
            weight=None,
            ignore_index=0,
            reduction='none',
            soft_label=True,
            dim=-1)

    def __call__(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        B, N = predicts.shape[:2]
        div = torch.tensor([N]).astype('float32')

        predicts = nn.functional.softmax(predicts, dim=-1)
        aggregation_preds = torch.sum(predicts, dim=1)
        aggregation_preds = torch.divide(aggregation_preds, div)

        length = batch[2].astype("float32")
        batch = batch[3].astype("float32")
        batch[:, 0] = torch.subtract(div, length)
        batch = torch.divide(batch, div)

        loss = self.loss_func(aggregation_preds, batch)
        return {"loss_ace": loss}
