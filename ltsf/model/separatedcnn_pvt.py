import math
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .pvt import PyramidVisionTransformer

class spcnn(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

    

class separated(nn.Module):
    def __init__(self, n_layer, img_size, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios, qkv_bias, 
                qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, depths, sr_ratios, num_stages) -> None:
        super().__init__()

        in_channel = 2
        tmp_channel = 4
        out_channel = 8

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv3d(2, 2, kernel_size=(3,3,1)))
        self.convs.append(nn.Conv3d(2,2,kernel_size=(1,1,3), padding=(0,0,1)))
        self.convs.append(nn.GELU())

        w, h = img_size[0] - 2, img_size[1] - 2

        for n in range(n_layer - 1):
            self.convs.append(nn.Conv3d(in_channel, tmp_channel, kernel_size=(3,3,1)))
            self.convs.append(nn.Conv3d(tmp_channel, out_channel, kernel_size=(1,1,3), padding=(0,0,1)))
            self.convs.append(nn.GELU())
            in_channel = out_channel
            tmp_channel = in_channel*2
            out_channel = tmp_channel*2
            w, h = w-2, h-2

        self.conv2 = nn.Conv3d(in_channel, 1, kernel_size=(1,1,1))

        n_patch = math.gcd(w, h)
        w_, h_ = int(w / n_patch), int(h / n_patch)

        print(type(w), type(w_), type(h), type(h_))


        self.pvt = PyramidVisionTransformer(img_size = (w, h), patch_size = (w_, h_), in_chans = in_chans, num_classes = num_classes, 
                embed_dims = embed_dims, num_heads = num_heads, mlp_ratios = mlp_ratios, qkv_bias = qkv_bias, 
                qk_scale = qk_scale, drop_rate = drop_rate, attn_drop_rate = attn_drop_rate, drop_path_rate = drop_path_rate, 
                norm_layer = norm_layer, depths = depths, sr_ratios = sr_ratios, num_stages = num_stages)


    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = torch.squeeze(self.conv2(x), dim=1)
        x = rearrange(x, 'b w h t -> b t w h')
        x = self.pvt(x)
        return x

if __name__ == '__main__':
    w, h = 72,24
    for i in range(10):
        w, h = w-2, h-2
        n_patch = math.gcd(w,h)
        w_, h_ = w / n_patch, h / n_patch
        print(w, h, n_patch, w_, h_)