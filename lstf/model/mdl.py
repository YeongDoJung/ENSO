import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, repeat

import sys
sys.path.append('.')

from .model import *
from ..utils import metric

class RFB_Transformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Model_3D, self).__init__()

        self.rfb1 = RFB(in_channels, out_channels)
        self.rfb2 = RFB(out_channels, out_channels*2)
        self.rfb3 = RFB(out_channels*2, out_channels*4) # (n, 64, 9, 3, 3)

        self.transformer = nn.Transformer(d_model = 64, batch_first=True) #embed_dim must be divisible by num_heads

        self.dense_1 = nn.Linear(d_model, 1)
        self.dense_2 = nn.Linear(81, 1)

        self.softmax = nn.softmax()

        self.PE = nn.Parameter(torch.rand(1, 81, 64))



    def forward(self, x) :
        b, n, c = x.shape[0]
        #feature extract
        out = self.rfb1(x)
        out = F.MCDropout(out) # MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)

        out = self.rfb2(out)
        out = F.MCDropout(out) # MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)
        
        out = self.rfb3(out)
        out = F.MCDropout(out) # MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)

        #Positional Encoding
        out = rearrange(out, 'n f c w h -> n (cwh) f')
        self.PE = repeat(self.PE, '() n c -> b n c', b = b)
        out += out + self.PE

        out = self.transformer(out)
        out = rearrange(out, 'n w h -> n (wh)')

        #prediction
        out = self.dense(out)
        out = self.softmax(out)

        return out
