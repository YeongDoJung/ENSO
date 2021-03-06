from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from . import vit

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys
import math

sys.path.append('.')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class res_enc(nn.Module):
    def __init__(self, in_channels = 2, d_model =  1024, depth = 24, heads = 16, dim_head = 64, mlp_dim = 1024):
        super(res_enc, self).__init__()

        self.resnet = models.resnet152(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7,7), stride=2)
        in_feature_num = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feature_num, d_model)

        # depth = num of encoder stack / heads, dim_head = attention head # & inner dim / mlp_dim = feed-forward inner dim
        self.vit = vit.ViT(num_classes=128, dim=d_model, depth = 24, heads = 32, mlp_dim = 1024)

        # self.act = nn.Sigmoid()
        self.relu = nn.ReLU()

        # self.PE = nn.Parameter(torch.rand(1, 3, d_model))
        self.pe = PositionalEncoding(d_model = d_model)

        self.dense = nn.Linear(128, 23)


    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x) :
        batch_size = x.shape[0]
        #feature extract
        x = rearrange(x, 'b t c w h -> (b t) c w h')
        out = self.resnet(x)
        out = rearrange(out, '(b_o t) f -> b_o t f', b_o = batch_size, t = 3)

        #Positional Encoding

        # pe = repeat(self.PE, '() c f -> b c f', b = batch_size)
        out = self.pe(out)

        out = self.vit(out)

        out = torch.squeeze(self.dense(out))

        return out

if __name__ == "__main__":
    toy = torch.zeros([48,3,2,72,24])
    toy_tgt = torch.zeros([48,23,512])
    model = res_vit()
    out = model(toy)
    print(out.shape)