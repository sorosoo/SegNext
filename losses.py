import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
        weight: 每一种类别的权重，越大，说明该类别越重要
                [weight_1, weight_2, ...]
                len(weight) = classes_num
        gamma: 为0表示关闭该参数的影响，如果需要使用，范围应为(0.5, 10.0)
    """
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, x, y):
        logp = self.ce(x, y)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()




if __name__ == "__main__":
    pass