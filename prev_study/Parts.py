from numpy.random.mtrand import set_state
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

import dataprocessing as dp
import h5py
import random
import convlstm
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def scaling(array) :
    scaler = MaxAbsScaler()
    scaler.fit(array)
    return scaler.transform(array)

class ConvLSTM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLSTM_Block, self).__init__()
        self.convlstm = convlstm.ConvLSTM(
                input_dim=in_channels,
                hidden_dim=out_channels,
                kernel_size=(3, 3),
                num_layers=1,
                batch_first=True,
            )
        self.leakyrelu = nn.Sequential(nn.LeakyReLU())

    def forward(self, x):
        _, out = self.convlstm(x)
        # print(out)
        # print(out[0][0].shape)
        out = self.leakyrelu(out[0][0])
        return out

def MCDropout(act_vec, p=0.5, apply=True):
    return F.dropout(act_vec, p=p, training=apply)

class Conv3d_Block_prof(nn.Module):
    def __init__(self, in_channels, out_channels, drop, tKernel = 1, downstride=(2, 2, 2)):
        super(Conv3d_Block_prof, self).__init__()
        # self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

        # (𝑊−𝐹+2𝑃)/𝑆+1 = (3 - 2 + 2p)/1 + 1 
        tPadding = tKernel - 1
        gain = 1.0
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), stride = (1, 1, 1), padding = (1, 1, 0), bias=False)
        # nn.init.kaiming_normal_(self.conv.weight)
        # nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv1.weight, gain=gain)
        # nn.init.zeros_(self.conv1.bias)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride = downstride, padding = (1, 1, 1), bias=False)
        # nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.orthogonal_(self.conv2.weight, gain=gain)
        # nn.init.zeros_(self.conv2.bias)

        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride = downstride, padding = (1, 1, 1), bias=False)
        # nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.orthogonal_(self.conv3.weight, gain=gain)
        # nn.init.zeros_(self.conv3.bias)

        self.droprate = drop

    def forward(self, x, applyDropout=True):
        # Residual learning
        out = self.relu(x)
        out = self.conv1(out)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.conv2(out)
        out = MCDropout(out, self.droprate, apply=True)

        # input forward
        out2 = self.conv3(x)
        out2 = MCDropout(out2, self.droprate, apply=True)

        out += out2
        return out

class Conv3d_Block_prof_condition(nn.Module):
    def __init__(self, in_channels, out_channels, drop, tKernel = 1, downstride=(2, 2, 2)):
        super(Conv3d_Block_prof_condition, self).__init__()
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

        # (𝑊−𝐹+2𝑃)/𝑆+1 = (3 - 2 + 2p)/1 + 1 
        tPadding = tKernel - 1
        gain = 1.0
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride = (1, 1, 1), padding = (1, 1, 1), bias=True)
        # nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv1.weight, gain=gain)
        nn.init.zeros_(self.conv1.bias)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride = downstride, padding = (1, 1, 1), bias=True)
        # nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.orthogonal_(self.conv2.weight, gain=gain)
        nn.init.zeros_(self.conv2.bias)

        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride = downstride, padding = (1, 1, 1), bias=True)
        # nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.orthogonal_(self.conv3.weight, gain=gain)
        nn.init.zeros_(self.conv3.bias)

        self.droprate = drop

    def forward(self, x, applyDropout=True):
        # Residual learning
        out = self.relu(x)
        out = self.conv1(out)
        out = MCDropout(out, self.droprate, apply=True)
        out = self.relu(out)
        out = self.conv2(out)
        out = MCDropout(out, self.droprate, apply=True)

        # input forward
        out2 = self.conv3(x)
        out2 = MCDropout(out2, self.droprate, apply=True)

        out += out2
        return out


class Conv3d_Block_prof_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3d_Block_prof_CBAM, self).__init__()
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride = (1, 1, 1), padding = 1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv3d(out_channels, out_channels , kernel_size=(3, 2, 2), stride = (1, 2, 2), padding = 1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = nn.Conv3d(out_channels, out_channels , kernel_size=(3, 2, 2), stride = (1, 2, 2), padding = 1, bias=True)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.cbam = CBAM(out_channels, 8)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.cbam(out)

        out2 = self.conv3(x)
        out += out2
        return out

class Conv3d_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3d_Block, self).__init__()
        self.conv3d_normal1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True,)
        nn.init.kaiming_normal_(self.conv3d_normal1.weight)
        self.conv3d_normal2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3d_normal2.weight)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.conv = nn.Sequential(
            self.conv3d_normal1,
            self.batchnorm,
            nn.LeakyReLU(inplace=True),
            self.conv3d_normal2,
            self.batchnorm,
            nn.LeakyReLU(inplace=True),
            self.conv3d_normal2,
            self.batchnorm,
            )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        nn.LeakyReLU(inplace=True)
        return out

class Conv3d_Block2(nn.Module):
    def __init__(self, in_channels, out_channels, expansions=1):
        super(Conv3d_Block2, self).__init__()
        # self.conv3d_normal1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        # nn.init.kaiming_normal_(self.conv3d_normal1.weight)
        # self.conv3d_normal2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        # nn.init.kaiming_normal_(self.conv3d_normal2.weight)
        # self.conv3d_half = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 2, 2), stride=(1, 2, 2), padding=1, bias=True)
        # self.conv3d_half2 = nn.Conv3d(in_channels, out_channels, kernel_size = (3, 2, 2), stride=(1, 2, 2), padding=1, bias=True)
        # self.batchnorm = nn.BatchNorm3d(out_channels)
        # self.conv = nn.Sequential(
        #     self.conv3d_normal1,
        #     self.batchnorm,
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     self.conv3d_normal2,
        #     self.batchnorm,
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     self.conv3d_normal2,
        #     self.batchnorm,
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     self.conv3d_half
        #     )
        # def forward(self, x):
        #     out = self.conv(x)
        #     residual = self.conv3d_half2(x)
        #     out += residual
        #     return out
        self.expansions = expansions
        self.conv1 = conv1x1(in_channels, out_channels * expansions)
        self.conv2 = conv3x3(out_channels * expansions, out_channels * expansions)
        self.conv3 = conv1x1(out_channels * expansions, in_channels)
        self.batchnorm1 = nn.BatchNorm3d(out_channels * expansions)
        self.batchnorm2 = nn.BatchNorm3d(out_channels * expansions)
        self.batchnorm3 = nn.BatchNorm3d(in_channels)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x) :
        identity = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.batchnorm3(out)
        out += identity
        out = self.leakyrelu(out)

        return out


class Conv3d_convlstm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3d_convlstm, self).__init__()
        self.conv3d_normal = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3d_normal.weight)
        self.conv3d_half1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3d_half1.weight)
        self.conv3d_half2 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3d_half2.weight)

        self.conv = nn.Sequential(
            self.conv3d_normal,
            nn.Tanh(),
            self.conv3d_half1,
            nn.Dropout(p=0.5)
            )

    def forward(self, x):
        out = self.conv(x)
        residual = self.conv3d_half2(x)
        out += residual
        return out


class Conv3d_cbam(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3d_cbam, self).__init__()
        self.conv3d_normal = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv3d_normal.weight)
        self.conv3d_glorot = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv3d_glorot.weight)
        self.conv3d_half = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 2, 2), stride=(1, 2, 2), padding=1, bias=True)
        self.cbam = CBAM(out_channels, 16)
        self.conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            self.conv3d_normal,
            nn.LeakyReLU(inplace=True),
            self.conv3d_glorot,
            self.conv3d_half
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.cbam(out)
        residual = self.conv3d_half(x)
        out += residual
        return out

# class Conv2d_cbam(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Conv2d_cbam, self).__init__()
#         self.conv2d_normal = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
#         nn.init.kaiming_normal_(self.conv2d_normal.weight)
#         self.conv2d_glorot = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
#         nn.init.xavier_uniform_(self.conv2d_glorot.weight)
#         self.conv2d_half = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride=2, padding=1, bias=True)
#         self.cbam = CBAM_2D(out_channels, 16)
#         self.conv = nn.Sequential(
#             nn.LeakyReLU(inplace=True),
#             self.conv2d_normal,
#             nn.LeakyReLU(inplace=True),
#             self.conv2d_glorot,
#             self.conv2d_half
#             )

    def forward(self, x):
        out = self.conv(x)
        out = self.cbam(out)
        residual = self.conv2d_half(x)
        out += residual
        return out

class Conv2d_Basic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d_Basic, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=1, stride=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 2), padding=1, stride=2, bias=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        nn.init.kaiming_normal_(self.conv2.weight)
        # self.sequential = nn.Sequential(
        #     self.conv,
        #     nn.Tanh(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout(p=0.5)
        # )
        self.sequential = nn.Sequential(
            self.conv,
            self.batchnorm,
            nn.LeakyReLU(inplace=True),
            self.conv2,
            self.batchnorm,
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.sequential(x)
        return out


class Conv3d_Basic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv3d_Basic, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 2, 2), stride=(1, 2, 2), padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.batchnorm = nn.BatchNorm3d(out_channels)

        # self.sequential = nn.Sequential(
        #     self.conv,
        #     nn.Tanh(),
        #     self.conv2,
        #     nn.Dropout(p=0.5)
        # )
        self.sequential = nn.Sequential(
            self.conv,
            self.batchnorm,
            nn.LeakyReLU(inplace=True),
            self.conv2,
            self.batchnorm,
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )
    def forward(self, x):
        out = self.sequential(x)
        return out

class Avgpool(nn.Module):
    def __init__(self, size):
        super(Avgpool, self).__init__()
        self.avgpool = nn.Sequential(nn.AvgPool2d(size))

    def forward(self, x):
        out = self.avgpool(x)
        return out

class Avgpool_3D(nn.Module):
    def __init__(self, size):
        super(Avgpool_3D, self).__init__()
        self.avgpool = nn.Sequential(nn.AvgPool3d(kernel_size=(3, size, size), stride=(1, size, size), padding=1))
    def forward(self, x):
        out = self.avgpool(x)
        return out

##############################################################
## Data Loader
##############################################################

class datasets_general_3D(Dataset):
    def __init__(self, SSTFile, SSTFile_label, lead, sstName, hcName, labelName, conditioned = False):
        sstData =  dp.ReadData(SSTFile)
        sst = sstData[sstName][:, :, :, :]
        sst = np.expand_dims(sst, axis = 0)

        hc = sstData[hcName][:, :, :, :]
        hc = np.expand_dims(hc, axis = 0)
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = np.transpose(tr_x, (1, 0, 4, 3, 2)) #(1, 35532, 3, 24, 72) -> (35532, 1, 72, 24, 3)
        tdim, _, _, _, _ = tr_x.shape

        sstData_label = dp.ReadData(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0]
        tr_y_c = np.zeros((tdim,12))
        for i in range(tdim):
            mod = i%12
            tr_y_c[i,mod] = 1

        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, lead-1])
        self.tr_y_c = np.array(tr_y_c)
        self.condition = conditioned

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        # return (self.tr_x[idx] - np.mean(self.tr_x[idx])) / np.std(self.tr_x[idx]), self.tr_y[idx], self.tr_y_c[idx]
        return self.tr_x[idx], self.tr_y[idx], self.tr_y_c[idx]

class datasets_general_3D_alllead_add(Dataset):
    def __init__(self, SSTFile, SSTFile_label, SSTFile2, SSTFile_label2, lead, sstName, hcName, labelName, noise = False):
        sstData =  dp.ReadData(SSTFile)
        sst = sstData[sstName][:, :, :, :]
        sst = np.expand_dims(sst, axis = 0)

        sstData2 =  dp.ReadData(SSTFile2)
        sst2 = sstData2[sstName][:, :, :, :]
        sst2 = np.expand_dims(sst2, axis = 0)
        sst = np.concatenate((sst, sst2), axis=1)
        print(sst.shape)

        hc = sstData[hcName][:, :, :, :]
        hc = np.expand_dims(hc, axis = 0)

        hc2 = sstData2[hcName][:, :, :, :]
        hc2 = np.expand_dims(hc2, axis = 0)
        hc = np.concatenate((hc, hc2), axis=1)

        tr_x = np.append(sst, hc, axis = 0)

        del sst, hc, sst2, hc2, sstData, sstData2

        tr_x = np.transpose(tr_x, (1, 0, 4, 3, 2))
        tdim, _, _, _, _ = tr_x.shape

        sstData_label = dp.ReadData(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0]
        sstData_label2 = dp.ReadData(SSTFile_label2)
        tr_y2 = sstData_label2[labelName][:, :, 0, 0]
        tr_y = np.concatenate((tr_y, tr_y2), axis=0)

        del tr_y2, sstData_label, sstData_label2

        tr_y_c = np.zeros((tdim,12))
        for i in range(tdim):
            mod = i%12
            tr_y_c[i,mod] = 1

        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, :])
        self.tr_y_c = np.array(tr_y_c)
        self.noise = noise

        self.mask = self.tr_x[0]
        self.mask[self.mask!= 0.0] = 1.0

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        if self.noise == True:
            # prev = idx -1
            # if (idx == 0 ):
            #     prev = 0
            # next = idx + 1
            # if (idx == len(self.tr_x)-1):
            #     next = len(self.tr_x)-1
            # x_p = self.tr_x[prev]
            # x_n = self.tr_x[next]

            x = self.tr_x[idx]
            # alpha = np.random.rand(1)/2.0
            # x[:, :, 0] = (alpha)*x_p[:, :, 0] + (1.0-alpha)*x_p[:, :, 1]
            # alpha = np.random.rand(1)/2.0
            # x[:, :, 2] = (1.0-alpha)*x_n[:, :, 1] + alpha*x_n[:, :, 2]
            myNoise = np.random.normal(loc=0, scale=0.2, size=self.tr_x[idx].shape)
            myNoise = myNoise*self.mask
            return x+myNoise, self.tr_y[idx, :], self.tr_y_c[idx]
            # return x, self.tr_y[idx, :], self.tr_y_c[idx]
        else:
            x = self.tr_x[idx]
            return x, self.tr_y[idx, :], self.tr_y_c[idx]
        # return (self.tr_x[idx] - np.mean(self.tr_x[idx])) / np.std(self.tr_x[idx]), self.tr_y[idx], self.tr_y_c[idx]

class datasets_general_3D_alllead(Dataset):
    def __init__(self, SSTFile, SSTFile_label, lead, sstName, hcName, labelName, noise = False):
        sstData =  dp.ReadData(SSTFile)
        sst = sstData[sstName][:, :, :, :]
        sst = np.expand_dims(sst, axis = 0)

        hc = sstData[hcName][:, :, :, :]
        hc = np.expand_dims(hc, axis = 0)
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = np.transpose(tr_x, (1, 0, 4, 3, 2)) #(2, 35532, 3, 24, 72) -> (35532, 2, 72, 24, 3)
        tdim, _, _, _, _ = tr_x.shape

        sstData_label = dp.ReadData(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0]
        tr_y_c = np.zeros((tdim,12))
        for i in range(tdim):
            mod = i%12
            tr_y_c[i,mod] = 1

        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, :])
        self.tr_y_c = np.array(tr_y_c)
        self.noise = noise

        self.mask = self.tr_x[0]
        self.mask[self.mask!= 0.0] = 1.0

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        if self.noise == True:
            # prev = idx -1
            # if (idx == 0 ):
            #     prev = 0
            # next = idx + 1
            # if (idx == len(self.tr_x)-1):
            #     next = len(self.tr_x)-1
            # x_p = self.tr_x[prev]
            # x_n = self.tr_x[next]

            x = self.tr_x[idx]
            # alpha = np.random.rand(1)/2.0
            # x[:, :, 0] = (alpha)*x_p[:, :, 0] + (1.0-alpha)*x_p[:, :, 1]
            # alpha = np.random.rand(1)/2.0
            # x[:, :, 2] = (1.0-alpha)*x_n[:, :, 1] + alpha*x_n[:, :, 2]
            # myNoise = np.random.normal(loc=0, scale=0.1, size=self.tr_x[idx].shape)
            # myNoise = myNoise*self.mask
            # return x+myNoise, self.tr_y[idx, :], self.tr_y_c[idx]
            return x, self.tr_y[idx, :], self.tr_y_c[idx]
        else:
            x = self.tr_x[idx]
            return x, self.tr_y[idx, :], self.tr_y_c[idx]
            # myNoise = np.random.normal(loc=0, scale=0.1, size=self.tr_x[idx].shape)
            # myNoise = myNoise*self.mask
            # return x+myNoise, self.tr_y[idx, :], self.tr_y_c[idx]
        # return (self.tr_x[idx] - np.mean(self.tr_x[idx])) / np.std(self.tr_x[idx]), self.tr_y[idx], self.tr_y_c[idx]
        
