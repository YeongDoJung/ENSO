import numpy as np
import netCDF4 as nc
from einops import rearrange, reduce

import torch
import torch.utils.data as D

class basicdataset(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        sstData =  nc.Dataset(SSTFile)
        sst = sstData[sstName][:, :, :, :]
        sst = np.expand_dims(sst, axis = 0)

        hc = sstData[hcName][:, :, :, :]
        hc = np.expand_dims(hc, axis = 0)
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = np.transpose(tr_x, (1, 0, 4, 3, 2)) #(2, 35532, 3, 24, 72) -> (35532, 2, 72, 24, 3)
        tdim, _, _, _, _ = tr_x.shape

        sstData_label = nc.Dataset(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0]

        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, :])

    def _batchsize(self):
        return self.tr_x.shape

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        x = self.tr_x[idx] 
        y = self.tr_y[idx, :]
        return x, y

class tgtdataset(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        sstData =  nc.Dataset(SSTFile)
        sst = sstData[sstName]
        sst = np.expand_dims(sst, axis = 0)

        hc = sstData[hcName]
        hc = np.expand_dims(hc, axis = 0)
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = rearrange(tr_x, 'c b t h w -> b c w h t') #(2, 35532, 3, 24, 72) -> (35532, 2, 72, 24, 3)

        sstData_label = nc.Dataset(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0]

        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, :])

    def _batchsize(self):
        return self.tr_x.shape

    def __len__(self):
        return len(self.tr_x) - 26

    def __getitem__(self, idx):
        x = self.tr_x[idx:idx+3, :, :, :, 0]
        shifted_right = self.tr_x[idx+1:idx+24, :, :, :, 0] 
        y = np.squeeze(self.tr_y[idx, :])
        return x, shifted_right, y

class tdimdataset(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        sstData =  nc.Dataset(SSTFile)
        sst = sstData[sstName][:, :, :, :]
        sst = np.expand_dims(sst, axis = 0)

        hc = sstData[hcName][:, :, :, :]
        hc = np.expand_dims(hc, axis = 0)
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = rearrange(tr_x, 'l n c h w -> n (l c) w h') #(2, 35532, 3, 24, 72) -> (35532, 6, 72, 24)

        sstData_label = nc.Dataset(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0]

        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, :])

    def _batchsize(self):
        return self.tr_x.shape

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        return self.tr_x[idx], self.tr_y[idx, :]

class rddataset(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        sstData =  nc.Dataset(SSTFile)
        sst = sstData[sstName][:, 0, :, :] # 35532, 24, 72
        sst = rearrange(sst, 'a b c -> 1 a c b')

        hc = sstData[hcName][:, 0, :, :]
        hc = rearrange(hc, 'a b c -> 1 a c b')
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = rearrange(tr_x, 'c n h w -> n c w h') #(2, 35532, 24, 72) -> (35532, 72, 24, 2)

        sstData_label = nc.Dataset(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0] # sstData_label.shpae = 35532, 23, 1, 1

        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, :])

    def _batchsize(self):
        return self.tr_x.shape

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        return self.tr_x[idx], self.tr_y[idx, :]
