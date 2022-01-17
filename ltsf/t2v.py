import torch
from torch import nn
import numpy as np
import math

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], 1)
'''
def oh_t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        print(tau.shape, w.shape)
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        print(tau.shape, w.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)

    print(v1.shape, tau.shape, w0.shape, b0.shape)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], 1)
'''

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        # if len(tau.shape) == 2 :
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
        # else :
        #     return oh_t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Model(nn.Module):
    def __init__(self, activation, in_feature_dim, hiddem_dim):
        super(Model, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(in_feature_dim, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(in_feature_dim, hiddem_dim)
        
        self.fc1 = nn.Linear(hiddem_dim, 81)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1,1)
        x = self.l1(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x
