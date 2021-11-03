import torch 
import torch.nn as nn
import numpy as np
from Parts import *
# torch.manual_seed(722)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        
        gain = 1.0
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        nn.init.orthogonal_(self.conv.weight, gain=gain)
        # nn.init.constant_(self.conv.bias.data, 0)
        # self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(3, 3, 1), dilation=(3, 3, 1))
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 5, 1), padding=(0, 2, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(5, 5, 1), dilation=(5, 5, 1))
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channel, out_channel, 1),
            BasicConv3d(out_channel, out_channel, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(7, 7, 1), dilation=(7, 7, 1))
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

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        gain = 1.0
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        nn.init.orthogonal_(self.encoder_att.weight, gain=gain)
        self.encoder_att.bias.data.fill_(0)

        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        nn.init.orthogonal_(self.decoder_att.weight, gain=gain)
        self.decoder_att.bias.data.fill_(0)

        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        nn.init.orthogonal_(self.full_att.weight, gain=gain)
        self.full_att.bias.data.fill_(0)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Model_3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer, num_answer, drop, input):
        super(Model_3D, self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        gain = 1.0

        self.rfb1 = RFB(in_channels, out_channels)
        self.rfb2 = RFB(out_channels, out_channels*2)
        self.rfb3 = RFB(out_channels*2, out_channels*4)

        # Regression
        encoder_dim = out_channels*4
        decoder_dim = num_layer
        attention_dim = num_layer
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.decode_step = nn.RNNCell(encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        nn.init.orthogonal_(self.decode_step.weight_hh, gain)
        nn.init.orthogonal_(self.decode_step.weight_ih, gain)
        self.decode_step.bias_ih.data.fill_(0)
        self.decode_step.bias_hh.data.fill_(0)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        nn.init.orthogonal_(self.init_h.weight, gain=gain)
        self.init_h.bias.data.fill_(0)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        nn.init.orthogonal_(self.f_beta.weight, gain=gain)
        self.f_beta.bias.data.fill_(0)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=drop)

        self.fc = nn.Linear(decoder_dim, 1)  # linear layer to find scores over vocabulary
        self.fc.bias.data.fill_(0)
        nn.init.orthogonal_(self.fc.weight, gain=gain)

        # Classification
        self.flatten = nn.Flatten()
        flattenSize =  5184#2592 #3456

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
        # out = self.conv(x)
        out = self.rfb1(x)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)

        out = self.rfb2(out)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)
        
        out = self.rfb3(out)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.maxpool2d(out)

        # Regression
        out = out.permute(0, 2, 3, 4, 1)
        encoder_dim = out.size(-1)
        batch_size = out.size(0)
        encoder_out = out.view(out.size(0), -1, encoder_dim)
        # print(encoder_out.shape)
        # inintial h and c
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        # c = self.init_c(mean_encoder_out)

        decode_lengths = 23
        predictions = torch.zeros(batch_size, decode_lengths, 1).to(device)
        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            # h, c = self.decode_step(attention_weighted_encoding, (h, c))
            h = self.decode_step(attention_weighted_encoding, h)
            preds = self.fc(self.dropout(h))
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
