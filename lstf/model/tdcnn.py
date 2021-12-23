import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.modules import padding

class Model2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout2d()

        self.layer1 = nn.Sequential(nn.Conv3d(2, 2, kernel_size=(8,4,1), padding=(4,2,0)),
                                    nn.Tanh(),
                                    nn.MaxPool3d((2,2,1), stride=(2,2,1)),
                                    nn.Dropout2d())
        
        self.layer2 = nn.Sequential(nn.Conv3d(2, 2, kernel_size=(4,2,1), padding=(2,1,0)),
                                    nn.Tanh(),
                                    nn.MaxPool3d((2,2,1), stride=(2,2,1)),
                                    nn.Dropout2d())

        self.layer3 = nn.Sequential(nn.Conv3d(2, 2, kernel_size=(4,2,1)),
                                    nn.Tanh())

        self.dense = nn.Linear(450,23)

    def forward(self, x):
        b, c, w, h, z = x.shape

        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = rearrange(x, 'b d w h c -> b (d w h c)')
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)
        x = self.dense(x)
        print(x.shape)

        return x 

if __name__ == '__main__':
    a = torch.rand([64,2,72,24,3])
    toy = Model_2D()
    out = toy(a)
    print(out)

