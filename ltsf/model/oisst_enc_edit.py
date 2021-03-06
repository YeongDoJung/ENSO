import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, MultiheadAttention, ModuleList, Dropout, Linear, LayerNorm, BatchNorm1d ,BatchNorm2d, BatchNorm3d
from torch.nn.init import xavier_uniform_

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys

sys.path.append('.')

class RFB_Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes = 23, d_model = 128, nhead = 8, num_encoder_layers = 6,
                 dim_feedforward = 2048, dropout = 0.1,
                 activation = F.gelu, layer_norm_eps = 1e-5):
        super(RFB_Transformer, self).__init__()

        self.rfb1 = RFB(in_channels, out_channels)
        self.rfb2 = RFB(out_channels, out_channels*2)
        self.rfb3 = RFB(out_channels*2, out_channels*4) 
        self.rfb4 = RFB(out_channels*4, out_channels*8) 


        self.transformer = Transformer(d_model = d_model, nhead = nhead, num_encoder_layers = num_encoder_layers,
                 dim_feedforward = dim_feedforward, dropout = dropout,
                 activation = activation, layer_norm_eps = layer_norm_eps, batch_first = False, norm_first = False)
        # depth = num of encoder stack / heads, dim_head = attention head # & inner dim / mlp_dim = feed-forward inner dim

        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool3d(kernel_size = (2, 2, 1), stride=(2, 2, 1))

        self.arr1 = Rearrange('n f c w h -> n (c w h) f')

        self.PE = nn.Parameter(torch.rand(1, 2970, 128))

        self.dense = nn.Linear(2970, num_classes)

    def forward(self, x) :
        b = x.shape[0]
        #feature extract
        out = self.maxpool(x)
        out = self.rfb1(x)
        out = self.gelu(out)
        # print('rfb1out',torch.sum(torch.isnan(out)))
        # out = F.dropout(out) # MCDropout(out, self.droprate, apply=True)

        out = self.maxpool(out)
        out = self.rfb2(out)
        out = self.gelu(out)
        # out = F.dropout(out) # MCDropout(out, self.droprate, apply=True)

        out = self.maxpool(out)
        out = self.rfb3(out)
        out = self.gelu(out)

        # out = F.dropout(out) # MCDropout(out, self.droprate, apply=True)
        out = self.maxpool(out)
        out = self.rfb4(out)
        out = self.gelu(out)
        
        out = self.arr1(out)

        pe = repeat(self.PE, '() n c -> b n c', b = b)
        out += out + pe

        out = self.transformer(out)

        out = torch.mean(out, dim = -1)

        out = self.dense(out)

        return out


class Transformer(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-4, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first


    def forward(self, src: Tensor) -> Tensor:

        memory = self.encoder(src)

        return memory

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:

        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            # output = rearrange(output, 'a b c -> a c b')
            output = self.norm(output)
            # output = rearrange(output, 'a b c -> a c b')

        return output

class TransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm1 = BatchNorm1d(d_model)
        # self.norm2 = BatchNorm1d(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)


    def forward(self, src: Tensor) -> Tensor:
        x = src
        x = self.norm1(x + self._sa_block(x))
        x = self.norm1(x + self._ff_block(x))
        # x = x + self._sa_block(x)
        # x = rearrange(x, 'a b c -> a c b')
        # x = self.norm1(x)
        # x = rearrange(x, 'a b c -> a c b')
        # x = x + self._ff_block(x)
        # x = rearrange(x, 'a b c -> a c b')
        # x = self.norm2(x)
        # x = rearrange(x, 'a b c -> a c b')
        return x


    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x,
                           need_weights=False)

        return self.dropout1(x[0])

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        
        gain = 1.0
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        xavier_uniform_(self.conv.weight)
        # nn.init.orthogonal_(self.conv.weight, gain=gain)
        # nn.init.constant_(self.conv.bias.data, 0)
        # self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 5, 1), padding=(0, 2, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
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


        x = x_cat + self.conv_res(x)
        return x

if __name__ == "__main__":
    toy = torch.rand([48,2,360,180,3])
    model = RFB_Transformer(2, 16)
    out = model(toy)
    print(out.shape)
    print(torch.isnan(out))