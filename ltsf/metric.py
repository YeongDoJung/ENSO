import sys

import numpy as np
import torch.nn as nn
import torch

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

    def custom(self, pr, gt):
        et = torch.abs(pr, gt)

    def forward(self, pr, gt):
        return mean_squared_error(pr, gt, multioutput='raw_values')


if __name__ == '__main__':
    a = torch.rand(430, 23)
    b = torch.rand(430, 23)
    loss = corrcoefloss()
    print(loss(a,b).shape, loss(a,b)[0][1])