import sys
from einops import rearrange

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.functional as F

from .util import stdout
from sklearn.metrics import mean_squared_error

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def CorrelationSkill(real, pred):
    corrAvg = 0
    corr = np.corrcoef(real, pred)[0][1]
    if np.isnan(corr):
        corr = 0
    corrAvg += corr
    return corrAvg

class PearsonLoss_old(nn.Module):
    def __init__(self):
        super(PearsonLoss_old, self).__init__()

    def forward(self, x, y):
        b = x.size(0)
        tmp = 0
        tt = 0
        for i in range(b):
            xb = x[i, :] - torch.mean(x[i, :])
            yb = y[i, :] - torch.mean(y[i, :])
            num = torch.sum(xb*yb)
            div = torch.sqrt(torch.sum((xb**2))*torch.sqrt(torch.sum(yb**2))) + 1e-4
            tt = (1 - num / div)
            if torch.isnan(tt):
                stdout(str(num) + ',' + str(div))
                tt = 1
            tmp += tt
        if torch.isnan(tmp/b):
            tmp = torch.tensor([1])
        else:
            tmp /= b

        return tmp

class corrcoefloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        tmp = torch.cat((x,y),dim=0)
        cov = torch.cov(tmp)
        coef = torch.corrcoef(cov)[0][1]
        if torch.isnan(coef):
            coef = torch.tensor(0)
        else:
            pass
        coef = torch.tensor(1) - coef

        return coef

class weightedMSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,y):
        device = x.device
        err = x - y
        a = torch.linspace(0.5, 1, 23).to(device=device)
        a = rearrange(a, '... -> 1 ...')
        werr = torch.mean(a * err**2)
        return werr

class twoloss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,y):
        device = x.device
        a = PearsonLoss_old(x, y)
        b = mse(x, y)
        return 0.5*a + 0.5*b

def pearson(x, y):
    ns = 1e-4
    dv = 0
    for i in range(x.shape[0]):
        xb = x[i] - np.mean(x[i])
        yb = y[i] - np.mean(y[i])
        print(xb, yb)
        num = np.cov(xb, yb)
        div = np.sum(np.sqrt(xb**2)*np.sqrt(yb**2)) + 1e-4
        ns += num
        dv += div

    tmp = 1 - ns / dv
    tmp /= x.shape[0]

    return tmp

# def pearson(pred, gt):
#     allLoss = 0
#     for i in range(pred.shape[0]):
#         score = pred[i, :]
#         target = gt[i, :]
#         vx = score - torch.mean(score)
#         vy = target - torch.mean(target)
#         add = torch.sum((score - target) ** 2) / pred.shape[1]
#         loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) 
#         allLoss += 1.0 - loss + add*0.5
#     allLoss /= pred.shape[0]
#     return allLoss

class mse(nn.Module):
    def __init__(self) -> None:
        super(mse, self).__init__()
        self.m = nn.MSELoss(reduction='mean')

    def custom(self, pr, gt):
        et = torch.abs(pr, gt)

    def forward(self, pr, gt):
        return self.m(pr + (1e-8), gt + (1e-8))

# Generalized Extreme Value Loss (Frechet GEVL + Gumbel GEVL)
def FrechetGEVL(pred, target, a=13, s=1.7):
    temp = torch.abs(pred - target) / s + np.power(a / (1 + a), 1 / a)
    return ((temp ** -a) + (1 + a) * torch.log(temp)).mean()

def GumbelGEVL(pred, target, r=1.1):
    return (torch.pow(1 - torch.exp(-torch.pow(pred - target, 2)), r) * torch.pow(pred - target, 2)).mean()

# Dynamic Shift (Extreme Value Loss + MemNN)
def ExtremeValueLoss(pred, target, proportion, r=1):
    # pred (batch_size, 3)      : left extreme, normal, right extreme 확률을 나타내는 3차원 신경망 출력
    # target (batch_size, 3)    : left extreme, normal, right extreme 중 하나를 나타내는 one-hot 벡터
    # proportion (3,)           : 순서대로 left extreme, normal, right extreme의 표본 개수를 담고 있는 3차원 벡터
    
    proportion = torch.from_numpy(proportion / np.sum(proportion)).cuda()
    return -((1 - proportion) * (torch.pow(1 - pred / r, r) * target * torch.log(pred + 1e-6) + torch.pow(1 - (1 - pred) / r, r) * (1 - target) * torch.log(1 - pred + 1e-6))).mean()

class FGELV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = 13
        self.s = 1.7

    def forward(self, pred, target):
        temp = torch.abs(pred - target) / self.s + np.power(self.a / (1 + self.a), 1 / self.a)
        return ((temp ** -self.a) + (1 + self.a) * torch.log(temp)).mean()


class GGELV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r = 1.1
    def forward(self, pred, target):
        return (torch.pow(1 - torch.exp(-torch.pow(pred - target, 2)), self.r) * torch.pow(pred - target, 2)).mean()

if __name__ == '__main__':
    a = torch.rand(430, 23)
    b = torch.rand(430, 23)
    loss = corrcoefloss()
    print(loss(a,b).shape, loss(a,b)[0][1])
