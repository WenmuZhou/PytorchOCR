import torch


class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list=None, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, batch=None, *args, **kwargs):
        if 'res' in preds:
            preds = preds['res']
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        label_list = self.label_list
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        pred_idxs = preds.argmax(axis=1)
        decode_out = [(label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)]
        if batch is None:
            return decode_out
        label = [(label_list[idx], 1.0) for idx in batch[1].cpu().numpy()]
        return decode_out, label
