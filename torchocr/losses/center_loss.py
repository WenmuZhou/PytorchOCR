import os
import pickle

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.randn(self.num_classes, self.feat_dim, dtype=torch.float)

        if center_file_path is not None:
            assert os.path.exists(
                center_file_path
            ), f"center path({center_file_path}) must exist when it is not None."
            with open(center_file_path, 'rb') as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = torch.from_numpy(char_dict[key])

    def __call__(self, predicts, batch):
        assert isinstance(predicts, (list, tuple))
        features, predicts = predicts

        feats_reshape = torch.reshape(
            features, [-1, features.shape[-1]]).astype("float64")
        label = torch.argmax(predicts, dim=2)
        label = torch.reshape(label, [label.shape[0] * label.shape[1]])

        batch_size = feats_reshape.shape[0]

        #calc l2 distance between feats and centers  
        square_feat = torch.sum(torch.square(feats_reshape),dim=1, keepdim=True)
        square_feat = square_feat.expand([batch_size, self.num_classes])
        square_center = torch.sum(torch.square(self.centers), dim=1, keepdim=True)
        square_center = square_center.expand([self.num_classes, batch_size]).astype(torch.float64)
        square_center = torch.permute(square_center, [1, 0])

        distmat = torch.add(square_feat, square_center)
        feat_dot_center = torch.matmul(feats_reshape, torch.permute(self.centers, [1, 0]))
        distmat = distmat - 2.0 * feat_dot_center

        #generate the mask
        classes = torch.arange(self.num_classes, dtype=torch.int)
        label = torch.unsqueeze(label, 1).expand((batch_size, self.num_classes))
        mask = torch.equal(classes.expand([batch_size, self.num_classes]), label).astype(torch.float64)
        dist = torch.multiply(distmat, mask)

        loss = torch.sum(torch.clip(dist, min=1e-12, max=1e+12)) / batch_size
        return {'loss_center': loss}
