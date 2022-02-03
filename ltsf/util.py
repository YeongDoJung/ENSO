import torch
from torch import nn
import numpy as np
import sys
import matplotlib.pyplot as plt

def subsequent_mask(size):
    atten_shape = (1, size, size)
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8') # masking with upper triangle matrix
    return torch.from_numpy(mask)==0 # reverse (masking=False, non-masking=True)

def make_std_mask(tgt):
    tgt_mask = (tgt != np.zeros_like(tgt)) # pad masking
    tgt_mask = tgt.unsqueeze(-2) # reshape (n_batch, seq_len) -> (n_batch, 1, seq_len)
    tgt_mask = tgt_mask & torch.tensor(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) # pad_masking & subsequent_masking
    return tgt_mask

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

def stdout(ss):
    sys.stdout.write(ss + '\r')
    sys.stdout.flush()

def ploter(a, fp):
    timeline = np.arange(0, 23)

    plt.plot(timeline, np.sqrt(a), marker='', color='blue', linewidth=1)
    plt.legend()
    plt.savefig(fp, orientation='landscape', bbox_inches='tight')