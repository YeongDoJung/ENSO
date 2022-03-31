from base64 import encode
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from einops import rearrange

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class MemNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, memory_size, time_step):
        super(MemNN, self).__init__()

        # lstm module
        self.forward1 = nn.Sequential(
                TimeDistributed(nn.Linear(hidden_size, hidden_size)),
                TimeDistributed(nn.BatchNorm1d(hidden_size)),
                nn.ReLU(inplace=True),
                TimeDistributed(nn.Linear(hidden_size, output_size)))
        
        self.fc    = TimeDistributed(nn.Linear(1, hidden_size))
        self.lstm  = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        self.shift = nn.Parameter(torch.zeros(1,))

        # memory module
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
        self.S   = nn.Parameter(torch.zeros(time_step, memory_size, hidden_size), requires_grad=True)
        self.q   = nn.Parameter(torch.zeros(memory_size, time_step, 3), requires_grad=True)

    def construct_memory(self, encoder_x, decoder_x, y):
        # encoder_x (batch_size, time_step, 1) : encoder input sequence
        # decoder_x (batch_size, time_step, 1) : decoder input sequence (zero-filled vector)
        # y (batch_size, 3)                    : left extreme, normal, right extreme 중 하나를 나타내는 one-hot 벡터
    
        # x = encoder_x.permute(1, 0, 2) # (batch_size, time_step, channel) → (time_step, batch_size, channel)
        x = rearrange(encoder_x, 'a b c ... -> b a c ...')
        x = self.fc(x)

        x, h = self.gru(x)

        if decoder_x is None:
            decoder_x = torch.zeros_like(encoder_x)

        x = decoder_x.permute(1, 0, 2)
        x = self.fc(x)
        x = self.gru(x, h)[0]

        self.S = nn.Parameter(x, requires_grad=False)
        self.q = nn.Parameter(y, requires_grad=False)

    def forward(self, encoder_x, decoder_x=None):
        # x = encoder_x.permute(1, 0, 2) # (batch_size, time_step, channel) → (time_step, batch_size, channel)
        x = rearrange(encoder_x, 'a b ... -> b a ...')
        print(x.shape)
        x = self.fc(x)

        # estimator u (encoder)
        u = self.gru(x)[1]

        x, (h, s) = self.lstm(x)

        if decoder_x is None:
            decoder_x = torch.zeros_like(encoder_x)

        x = decoder_x.permute(1, 0, 2)
        x = self.fc(x)

        # estimator u (decoder) ################################################################################
        c = torch.matmul(self.gru(x, u)[0], self.S.transpose(1, 2))
        a = F.softmax(c, dim=-1)
        u = torch.sum(a.unsqueeze(3) * self.q.transpose(0, 1).unsqueeze(1), dim=-2).transpose(0, 1).contiguous()
        ########################################################################################################

        x = self.lstm(x, (h, s))[0]
        x = self.forward1(x).transpose(0, 1)

        return x + self.shift * torch.sum(u * torch.from_numpy(np.array([-1, 0, 1])).cuda(), dim=-1, keepdim=True), u


class memorynn(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, memory_size, time_step, eps) -> None:
        super().__init__()

        self.eps = eps
        
    def embedding_module(self):
        pass

    def isextreme(self):
        pass

    def attention(self):
        pass

    def forward(self, x):
        
        x = self.embedding_module(x)
        ev = self.isextreme(x)
        x = self.attention(x)



        pass
    
