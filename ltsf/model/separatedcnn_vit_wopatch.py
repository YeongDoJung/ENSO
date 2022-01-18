import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .vit_wo_patch import ViT

class separated_wopatch(nn.Module):
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
            in_channel = out_channel
            tmp_channel = in_channel*2
            out_channel = tmp_channel*2
            w, h = w-2, h-2

        self.conv2 = nn.Conv3d(in_channel, 1, kernel_size=(1,1,1))
        self.dense = nn.Linear(w*h, 512)
        

        self.vit = ViT(num_classes=23, dim=512, depth=12, heads=12, mlp_dim=2048, channels=3, dim_head=1024)


    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = self.conv2(x)
        x = rearrange(x, 'b 1 w h t -> b t (w h 1)')
        x = self.dense(x)
        x = self.vit(x)
        return x

if __name__ == '__main__':
    model = separated_wopatch(n_layer=3)
    toy = torch.rand([100,2,72,24,3])
    out = model(toy)