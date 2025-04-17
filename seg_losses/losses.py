import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def IOULoss(input, target, weight=None):

    epsilon = 1e-5

    num = target.size(0)
    input = torch.sigmoid(input)
    input = input.view(num, -1)
    target = target.view(num, -1)
    target = target.float()

    intersect = (input * target).sum(1)
    if weight is not None:
        intersect = weight * intersect

    union = (input + target).sum(1) - intersect

    return 1. - torch.mean(intersect / union.clamp(min=epsilon))


def DICELoss(input, target, weight=None, dimention=2):

    epsilon = 1e-5

    input = torch.sigmoid(input)
    input = input.view(-1)
    target = target.view(-1)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum()
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    if dimention == 2:
        denominator = (input * input).sum() + (target * target).sum()
    else:
        denominator = (input + target).sum()

    return 1. - 2 * (intersect / denominator.clamp(min=epsilon))


def KLDivLoss(input, target):
    epsilon = 1e-5

    n = input.size(0)
    input = F.softmax(input.view(n, -1), -1)
    target = target.float()
    target = target.view(n, -1)
    add = target.sum(-1)[:,None]
    target = torch.div(target, add.clamp(min=epsilon))

    return F.kl_div(torch.log(input), target, reduction='batchmean')


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            # 将alpha转换为tensor，以便在正确的设备上使用
            self.alpha = torch.tensor(self.alpha).reshape(2)
        elif isinstance(self.alpha, (float, int)):
            self.alpha = torch.tensor([self.alpha, 1.0 - self.alpha])
        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        # 确保alpha在正确的设备上
        device = output.device
        alpha = self.alpha.to(device) if isinstance(self.alpha, torch.Tensor) else torch.tensor(self.alpha, device=device)
        
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        device = inputs.device
        # 直接在inputs的设备上创建张量
        targets_one_hot = torch.zeros(log_probs.size(), device=device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss