from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np


class BCELoss(nn.Module, ABC):
    def __init__(self, alpha):
        super(BCELoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()
        self.alpha = alpha

    def forward(self, epoch, output, target):
        device = output.device
        output_tp, target_tp, output_fp, target_fp, output_nolabel, target_nolabel = [], [], [], [], [], []
        for idx, t in enumerate(target):
            if t[0] == 1:
                output_tp.append(output[idx])
                target_tp.append(t[1:])
            if t[0] == 0:
                output_fp.append(output[idx])
                target_fp.append(t[1:])
            if t[0] == 2:
                output_nolabel.append(output[idx])
                target_nolabel.append(t[1:])

        # true positive bce loss
        if output_tp.__len__() > 0:
            output_tp = torch.stack(output_tp).to(device)
            target_tp = torch.stack(target_tp).to(device)
            tp_loss = self.bce_loss(output_tp, target_tp)
        else:
            tp_loss = torch.zeros(1).to(device)

        # false positive l1 loss
        if output_fp.__len__() > 0:
            output_fp = torch.stack(output_fp).to(device)
            target_fp = torch.stack(target_fp).to(device)
            fp_loss = (output_fp * target_fp).sum() / output_fp.shape[0]
        else:
            fp_loss = torch.zeros(1).to(device)

        # no label bce loss
        if output_nolabel.__len__() > 0:
            output_nolabel = torch.stack(output_nolabel).to(device)
            target_nolabel = torch.stack(target_nolabel).to(device)
            nolabel_loss = self.bce_loss(output_nolabel, target_nolabel)
        else:
            nolabel_loss = torch.zeros(1).to(device)

        # print(tp_loss.item())
        # print(fp_loss.item())
        # alpha = 0.9
        # print(nolabel_loss)

        if epoch < 30:
            ratio = 0
        else:
            ratio = np.min((np.floor((epoch - 30) / 10) / 10, 0.4))

        return (0.8 - ratio) * tp_loss + ratio * nolabel_loss + 0.2 * fp_loss


class CosineMarginLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, m=0.35, s=64):
        super(CosineMarginLoss, self).__init__()
        self.w = nn.Parameter(torch.randn(embed_dim, num_classes))
        self.num_classes = num_classes
        self.m = m
        self.s = s

    def forward(self, output, label):
        x_norm = output / torch.norm(output, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)

        label_one_hot = F.one_hot(label.view(-1), self.num_classes).float() * self.m
        value = self.s * (xw_norm - label_one_hot)
        return F.cross_entropy(input=value, target=label.view(-1))