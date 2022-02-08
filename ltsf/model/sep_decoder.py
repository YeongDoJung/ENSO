import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys

sys.path.append('.')

class sep_dec(nn.Module):
    def __init__(self):
        super(sep_dec, self).__init__()

        in_channel = 4
        out_channel = 8

        n_layer = 5

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv3d(2, 4, kernel_size=(3,3,1), padding=(1,1,0)))
        self.convs.append(nn.Conv3d(4, 4, kernel_size=(1,1,3), padding=(0,0,1)))
        self.convs.append(nn.GELU())
        self.convs.append(nn.Dropout3d())

        for n in range(n_layer - 1):
            self.convs.append(nn.Conv3d(in_channel, out_channel, kernel_size=(3,3,1), padding=(1,1,0)))
            self.convs.append(nn.Conv3d(out_channel, out_channel, kernel_size=(1,1,3), padding=(0,0,1)))
            self.convs.append(nn.GELU())
            self.convs.append(nn.Dropout3d())
            in_channel = out_channel
            out_channel = in_channel*2

        self.conv2 = nn.Conv3d(in_channel, 1, kernel_size=(1,1,1))

        self.transformer = nn.Transformer(d_model = 72*24, nhead = 12, batch_first=True) #embed_dim must be divisible by num_heads

        # depth = num of encoder stack / heads, dim_head = attention head # & inner dim / mlp_dim = feed-forward inner dim
        self.arr1 = Rearrange('n c w h f -> n f (c w h)')

        self.PE = nn.Parameter(torch.rand(1, 3, 72*24))

        self.dense = nn.Linear(72*24, 1)

    def forward(self, x) :
        b, device = x.shape[0], x.device

        out = x[:, :, :, :, :3]
        out_y = x[:, :, :, :, 1:]

        for layer in self.convs:
            out = layer(out)
            out_y = layer(out_y)
        out = self.conv2(out)
        out_y = self.conv2(out_y)

        #Positional Encoding
        out = self.arr1(out)
        out_y = self.arr1(out_y)

        pe = repeat(self.PE, '() n c -> b n c', b = b)
        pe_y = repeat(self.PE, '() n c -> b n c', b = b)
        out += out + pe
        out_y = out_y + pe_y
        # out = rearrange(out, 'b n c -> b c n')

        #prediction

        some_spaces = torch.zeros([b, 23, 1728]).to(device=device)
        shifted_right = torch.cat([out_y, some_spaces], axis=1)

        out = self.transformer(out, shifted_right)

        out = torch.squeeze(self.dense(out), dim=-1)[:,3:]

        # out = self.act(out)

        return out
