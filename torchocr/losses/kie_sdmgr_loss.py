from torch import nn
import torch


class SDMGRLoss(nn.Module):
    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=0):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def pre_process(self, gts, tag):
        gts, tag = gts.numpy(), tag.numpy().tolist()
        temp_gts = []
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_gts.append(torch.tensor(gts[i, :num, :num + 1], dtype=torch.int64))
        return temp_gts

    def accuracy(self, pred, target, topk=1, thresh=None):
        """Calculate accuracy according to the prediction and target.

        Args:
            pred (torch.Tensor): The model prediction, shape (N, num_class)
            target (torch.Tensor): The target of each prediction, shape (N, )
            topk (int | tuple[int], optional): If the predictions in ``topk``
                matches the target, the predictions will be regarded as
                correct ones. Defaults to 1.
            thresh (float, optional): If not None, predictions with scores under
                this threshold are considered incorrect. Default to None.

        Returns:
            float | tuple[float]: If the input ``topk`` is a single integer,
                the function will return a single float as accuracy. If
                ``topk`` is a tuple containing multiple integers, the
                function will return a tuple containing accuracies of
                each ``topk`` number.
        """
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        if pred.shape[0] == 0:
            accu = [pred.new_tensor(0.) for i in range(len(topk))]
            return accu[0] if return_single else accu
        pred_value, pred_label = torch.topk(pred, maxk, dim=1)
        pred_label = pred_label.transpose(
            [1, 0])  # transpose to shape (maxk, N)
        correct = torch.equal(pred_label, (target.reshape([1, -1]).expand_as(pred_label)))
        res = []
        for k in topk:
            correct_k = torch.sum(correct[:k].reshape([-1]).astype('float32'),
                                   dim=0,
                                   keepdim=True)
            res.append(torch.multiply(correct_k, torch.tensor(100.0 / pred.shape[0])))
        return res[0] if return_single else res

    def forward(self, pred, batch):
        node_preds, edge_preds = pred
        gts, tag = batch[4], batch[5]
        gts = self.pre_process(gts, tag)
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].reshape([-1]))
        node_gts = torch.cat(node_gts)
        edge_gts = torch.cat(edge_gts)

        node_valids = torch.nonzero(node_gts != self.ignore).reshape([-1])
        edge_valids = torch.nonzero(edge_gts != -1).reshape([-1])
        loss_node = self.loss_node(node_preds, node_gts)
        loss_edge = self.loss_edge(edge_preds, edge_gts)
        loss = self.node_weight * loss_node + self.edge_weight * loss_edge
        return dict(
            loss=loss,
            loss_node=loss_node,
            loss_edge=loss_edge,
            acc_node=self.accuracy(
                torch.gather(node_preds, node_valids),
                torch.gather(node_gts, node_valids)),
            acc_edge=self.accuracy(
                torch.gather(edge_preds, edge_valids),
                torch.gather(edge_gts, edge_valids)))
