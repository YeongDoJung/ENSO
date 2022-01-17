import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.modules import padding

class Model2D(nn.Module):
    def __init__(self, num_conv=50, num_dense=50):
        super().__init__()

        self.dropout = nn.Dropout2d()

        self.layer1 = nn.Sequential(nn.Conv2d(6, num_conv, kernel_size=(8,4), padding=(4,2)),
                                    nn.Tanh(),
                                    nn.MaxPool2d((2,2), stride=(2,2)),
                                    nn.Dropout2d())
        
        self.layer2 = nn.Sequential(nn.Conv2d(num_conv, num_conv, kernel_size=(4,2), padding=(2,1)),
                                    nn.Tanh(),
                                    nn.MaxPool2d((2,2), stride=(2,2)),
                                    nn.Dropout2d())

        self.layer3 = nn.Sequential(nn.Conv2d(num_conv, num_conv, kernel_size=(4,2), padding=(2,1)),
                                    nn.Tanh())

        self.dense1 = nn.Linear(num_conv*19*7,num_dense)
        self.dense2 = nn.Linear(num_dense, 23)


    def forward(self, x):
        b, c, w, h = x.shape
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = rearrange(x, 'b w h c -> b (w h c)')
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x 

if __name__ == '__main__':
    a = torch.rand([400,6,72,24])
    toy = Model2D()
    out = toy(a)
    print(out)

