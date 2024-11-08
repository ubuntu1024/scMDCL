"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
import torch
import torch.nn as nn



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.7, cuda_device='0'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.cuda_device = cuda_device

    def forward(self, feature1,feature2=None, labels=None, mask=None):
        device = ("cuda:" + self.cuda_device if feature1.is_cuda else torch.device('cpu'))

        batch_size = feature1.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)+torch.eye(batch_size, dtype=torch.float32).to(device)
        if feature2 is not None:
            anchor_feature = feature1
            contrast_feature=feature2
        else:
            anchor_feature=feature1
            contrast_feature=feature1
        contrast_count = 1
        anchor_count = 1


        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        #logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        #logits = anchor_dot_contrast /(logits_max.detach())
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast/(logits_max.detach()-logits_min)



        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )  # 对角线设置成0


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 分母
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))   # 对应分子logits

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1)/ mask.sum(1)  # |1/P(i)|= mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
