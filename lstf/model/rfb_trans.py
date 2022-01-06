from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys

sys.path.append('.')

class RFB_Trf(nn.Module):
    def __init__(self, in_channels, out_channels, decoder = True, num_classes = 23, dim = 81, depth = 24, heads = 16, dim_head = 64, mlp_dim = 1024):
        super(RFB_Trf, self).__init__()

        self.rfb1 = RFB(in_channels, out_channels)
        self.rfb2 = RFB(out_channels, out_channels*2)
        self.rfb3 = RFB(out_channels*2, out_channels*4) # (n, 64, 9, 3, 3)

        self.conv1 = nn.Conv3d(in_channels=1,out_channels=1, kernel_size=(6,3,1), stride=(4,1,1))

        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,4,1))

        self.transformer = nn.Transformer(d_model = 64) #embed_dim must be divisible by num_heads

        # depth = num of encoder stack / heads, dim_head = attention head # & inner dim / mlp_dim = feed-forward inner dim

        # self.act = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size = (2, 2, 1), stride=(2, 2, 1))

        self.PE = nn.Parameter(torch.rand(1, 1, 64))

        self.tgt = nn.Parameter(torch.rand(23, 1, 64))

        self.dense = nn.Linear(64, 1)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x, tgt_mask: Optional[Tensor] = None) :
        batch_size = x.shape[0]
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

        out = rearrange(out, 'b d w h c -> b c w h d')

        outs = []

        for i in range(out.shape[1]):
            outs.append(self.conv1(torch.unsqueeze(out[:,i,:,:,:], axis=1)))
        
        out = torch.cat((outs[0], outs[1], outs[2]), dim=1)
        out = self.maxpool2(out)
        out = torch.squeeze(out)

        #Positional Encoding

        pe = repeat(self.PE, '() c f -> b c f', b = batch_size)
        out += pe

        out = rearrange(out, 'b c f -> c b f')

        tgt = repeat(self.tgt, 'f () c -> f b c', b = batch_size)

        out = self.transformer(out, tgt, tgt_mask = tgt_mask)

        out = self.dense(out)
        out = torch.squeeze(out)

        out = rearrange(out, 'c b -> b c')

        return out

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        
        gain = 1.0
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        nn.init.orthogonal_(self.conv.weight, gain=gain)
        # nn.init.constant_(self.conv.bias.data, 0)
        # self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(3, 3, 1), dilation=(3, 3, 1))
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 5, 1), padding=(0, 2, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(5, 5, 1), dilation=(5, 5, 1))
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(7, 7, 1), dilation=(7, 7, 1))
        )
        self.conv_cat = BasicConv3d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv3d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

if __name__ == "__main__":
    toy = torch.zeros([48,2,72,24,3])
    model = RFB_Transformer(2, 16)
    out = model(toy)
    print(out.shape)