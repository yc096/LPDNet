# _*_ coding: utf-8 _*_
# @Time : 2022/4/5 10:03 
# @Author : yc096
# @File : loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class structure_loss(nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def __call__(self, pred, mask):
        loss = self.structure_loss(pred, mask)
        return loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class structure_loss_x3(nn.Module):
    def __init__(self):
        super(structure_loss_x3, self).__init__()

    def __call__(self, pred, mask):
        p2 = self.structure_loss(pred[0], mask)
        p3 = self.structure_loss(pred[1], mask)
        p4 = self.structure_loss(pred[2], mask)
        loss = p2 + p3 + p4
        return loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class structure_loss_PraNet(nn.Module):
    def __init__(self):
        super(structure_loss_PraNet, self).__init__()

    def __call__(self, pred, mask):
        pd_loss = self.structure_loss(pred[0], mask)  # pd_loss
        ra5_loss = self.structure_loss(pred[1], mask)  # ra5_loss
        ra4_loss = self.structure_loss(pred[2], mask)  # ra4_loss
        ra3_loss = self.structure_loss(pred[3], mask)  # ra3_loss
        loss = pd_loss + ra5_loss + ra4_loss + ra3_loss  #
        return loss

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1e-5, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = torch.sigmoid(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        inter = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth
        loss = 1 - (2 * inter / union)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
