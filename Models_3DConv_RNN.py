import torch 
import torch.nn as nn
import numpy as np
from Parts import *
# torch.manual_seed(722)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model_3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer, num_answer, drop, input):
        super(Model_3D, self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        gain = 1.0
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding = (1, 1, 1), bias=True)
        nn.init.orthogonal_(self.conv.weight, gain=gain)
        self.conv.bias.data.fill_(0)

        self.conv2 = nn.Conv3d(out_channels, out_channels*2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding = (1, 1, 1), bias=True)
        nn.init.orthogonal_(self.conv2.weight, gain=gain)
        self.conv2.bias.data.fill_(0)

        self.conv3 = nn.Conv3d(out_channels*2, out_channels*4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding = (1, 1, 1), bias=True)
        nn.init.orthogonal_(self.conv3.weight, gain=gain)
        self.conv3.bias.data.fill_(0)

        # Regression
        encoder_dim = out_channels*4
        decoder_dim = num_layer
        attention_dim = num_layer

        self.decode_step = nn.RNNCell(encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        nn.init.orthogonal_(self.decode_step.weight_hh, gain)
        nn.init.orthogonal_(self.decode_step.weight_ih, gain)
        self.decode_step.bias_ih.data.fill_(0)
        self.decode_step.bias_hh.data.fill_(0)

        self.decode_step2 = nn.RNNCell(decoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        nn.init.orthogonal_(self.decode_step2.weight_hh, gain)
        nn.init.orthogonal_(self.decode_step2.weight_ih, gain)
        self.decode_step2.bias_ih.data.fill_(0)
        self.decode_step2.bias_hh.data.fill_(0)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        nn.init.orthogonal_(self.init_h.weight, gain=gain)
        self.init_h.bias.data.fill_(0)

        self.init_h2 = nn.Linear(decoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        nn.init.orthogonal_(self.init_h2.weight, gain=gain)
        self.init_h2.bias.data.fill_(0)

        self.f_beta = nn.Linear(encoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        nn.init.orthogonal_(self.f_beta.weight, gain=gain)
        self.f_beta.bias.data.fill_(0)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=drop)

        self.fc = nn.Linear(decoder_dim, 1)  # linear layer to find scores over vocabulary
        self.fc.bias.data.fill_(0)
        nn.init.orthogonal_(self.fc.weight, gain=gain)

        # Classification
        self.flatten = nn.Flatten()
        flattenSize =  5184 #896

        self.linear1 = nn.Linear(flattenSize, num_layer, bias=True)
        self.linear1.bias.data.fill_(0)
        nn.init.orthogonal_(self.linear1.weight, gain=gain)

        self.linear2 = nn.Linear(num_layer, 12, bias=True)
        self.linear2.bias.data.fill_(0)
        nn.init.orthogonal_(self.linear2.weight, gain=gain)

        self.droprate = drop
        self.num_layer = num_layer

        self.maxpool2d = nn.MaxPool3d(kernel_size = (2, 2, 1), stride=(2, 2, 1))
        # self.maxpool3d = nn.MaxPool3d(kernel_size = (2, 2, 2), stride=(2, 2, 2))

    def forward(self, x) :
        # print(x.shape)
        out = self.conv(x)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)

        out = self.conv2(out)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)
        
        out = self.conv3(out)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)

        # Regression
        out = out.permute(0, 2, 3, 4, 1)
        encoder_dim = out.size(-1)
        batch_size = out.size(0)
        encoder_out = out.view(out.size(0), -1, encoder_dim)
        # inintial h and c
        # print(encoder_out.shape)
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        h2 = self.init_h2(h)  # (batch_size, decoder_dim)

        decode_lengths = 23
        predictions = torch.zeros(batch_size, decode_lengths, 1).to(device)
        for t in range(decode_lengths):
            h = self.decode_step(mean_encoder_out, h)
            h2 = self.decode_step2(h, h2)
            preds = self.fc(self.dropout(h2))
            predictions[:, t, :] = preds

        # Classification
        flat = self.flatten(out)
        # print(flat.shape)
        out_c = self.linear1(flat)
        out_c = self.tanh(out_c)
        out_c = self.linear2(out_c)

        return (predictions, out_c)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
