# --------------------------------****************************-----------------------------------------------------
# ---------------The Engineering Source Code Of The BERT Fine Tune In Intention Recognition------------------------
# Copyright (C) 2020 Chongqing Normal university and Ma Shang Xiao Fei Financial
# This file is the code of focalloss
# focalloss is expounded in the link https://arxiv.org/abs/1708.02002
# the source code can found in https://github.com/unsky/
# --------------------------------****************************-----------------------------------------------------

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label[ignore] = 0

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = torch.ones_like(logits)
        mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        probs = torch.sigmoid(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(lb_one_hot == 1, probs, 1-probs)
        alpha = self.alpha*lb_one_hot + (1-self.alpha)*(1-lb_one_hot)
        loss = -alpha*((1-pt)**self.gamma)*torch.log(pt + 1e-12)
        loss[mask == 0] = 0
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum()/n_valid
        return loss