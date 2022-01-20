import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .vit import ViT

class spcnn(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

    

class separated(nn.Module):
    def __init__(self, n_layer) -> None:
        super().__init__()

        in_channel = 2
        tmp_channel = 4
        out_channel = 8

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv3d(2, 2, kernel_size=(3,3,1)))
        self.convs.append(nn.Conv3d(2,2,kernel_size=(1,1,3), padding=(0,0,1)))

        w, h = 70, 22

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


        self.vit = ViT(image_size=(w,h), patch_size=(w_,h_), num_classes=23, dim=512, depth=12, heads=12, mlp_dim=2048, channels=3, dim_head=1024)


    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = torch.squeeze(self.conv2(x), dim=1)
        x = rearrange(x, 'b w h t -> b t w h')
        x = self.vit(x)
        return x

if __name__ == '__main__':
    w, h = 72,24
    for i in range(10):
        w, h = w-2, h-2
        n_patch = math.gcd(w,h)
        w_, h_ = w / n_patch, h / n_patch
        print(w, h, n_patch, w_, h_)