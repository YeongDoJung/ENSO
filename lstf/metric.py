import numpy as np
import sklearn.metrics
import torch.nn as nn
import torch

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
    size = real.shape[0] // 12
    real = np.reshape(real, (size, 12))
    pred = np.reshape(pred, (size, 12))
    corrAvg = 0
    for i in range(12):
        corr = np.corrcoef(real[:, i], pred[:, i])[0][1]
        corrAvg += corr
    corrAvg = corrAvg / 12.0
    return corrAvg

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) #part of model, but it should not be considered a model parameter

    def forward(self, x):
        return self.pe[:, :x.size(1)]

def pearson(pred, gt):
    allLoss = 0
    for i in range(pred.shape[0]):
        score = pred[i, :]
        target = gt[i, :]
        vx = score - torch.mean(score)
        vy = target - torch.mean(target)
        add = torch.sum((score - target) ** 2) / pred.shape[1]
        loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) 
        allLoss += 1.0 - loss + add*0.5
    allLoss /= pred.shape[0]
    return allLoss