import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys

from .feature_extractor import RFB
sys.path.append('.')

class RFB_Transformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB_Transformer, self).__init__()

        self.rfb1 = RFB(in_channels, out_channels)
        self.rfb2 = RFB(out_channels, out_channels*2)
        self.rfb3 = RFB(out_channels*2, out_channels*4) # (n, 64, 9, 3, 3)

        self.transformer = nn.Transformer(d_model = 64) #embed_dim must be divisible by num_heads

        self.dense_1 = nn.Linear(64, 1)
        self.dense_2 = nn.Linear(81, 23)

        # self.act = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size = (2, 2, 1), stride=(2, 2, 1))

        self.arr1 = Rearrange('n f c w h -> n (c w h) f')

        self.PE = nn.Parameter(torch.rand(1, 81, 64))



    def forward(self, x) :
        b = x.shape[0]
        #feature extract
        out = self.rfb1(x)
        out = F.dropout(out) # MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.rfb2(out)
        out = F.dropout(out) # MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.rfb3(out)
        out = F.dropout(out) # MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool(out)

        #Positional Encoding
        out = self.arr1(out)

        pe = repeat(self.PE, '() n c -> b n c', b = b)
        out += out + pe

        tgt = torch.zeros_like(out)

        out = self.transformer(out, tgt)

        #prediction
        out = self.dense_1(out)
        out = out.squeeze(-1)
        out = self.dense_2(out)

        # out = self.act(out)

        return out
