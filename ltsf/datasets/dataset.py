from timeit import repeat
import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from einops import rearrange, reduce, repeat
from timm.models.registry import register_model

import torch
import torch.utils.data as D

@register_model
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

@register_model
class dtom(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        self.tr_x = np.load(SSTFile)

        self.tr_y = np.load(SSTFile_label)[:, :, 0, 0]

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        x = self.tr_x[idx] 
        y = self.tr_y[idx, :]
        return x, y

@register_model
class dtom_2d(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        self.tr_x = np.load(SSTFile)

        self.tr_y = np.load(SSTFile_label)[:, :, 0, 0]
        tr_x = rearrange(tr_x, 'a b c d e -> b (a c) d e')

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        x = self.tr_x[idx] 
        y = self.tr_y[idx, :]
        return x, y

@register_model
class tfdataset(D.Dataset):
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

        sstData_label = np.load(SSTFile_label)

        self.tr_y = sstData_label[:, :, 0]
        self.tr_x = np.array(tr_x)
        print(self.tr_x.shape, self.tr_y.shape)

    def __len__(self):
        return len(self.tr_x) - 26

    def __getitem__(self, idx):
        x = self.tr_x[idx+3] 
        y = self.tr_y[idx:idx+26,0]
        return x, y

@register_model
class tfhcnorm(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        sstData =  nc.Dataset(SSTFile)
        sst = sstData[sstName][:, :, :, :]
        sst = np.expand_dims(sst, axis = 0)

        hc = np.array(sstData[hcName][:, :, :, :])
        hc = (hc - hc.min()) / (hc.max() - hc.min())
        hc = np.expand_dims(hc, axis = 0)
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = np.transpose(tr_x, (1, 0, 4, 3, 2)) #(2, 35532, 3, 24, 72) -> (35532, 2, 72, 24, 3)
        tdim, _, _, _, _ = tr_x.shape

        sstData_label = np.load(SSTFile_label)

        self.tr_y = sstData_label[:, :, 0]
        self.tr_x = np.array(tr_x)
        print(self.tr_x.shape, self.tr_y.shape)

    def __len__(self):
        return len(self.tr_x) - 26

    def __getitem__(self, idx):
        x = self.tr_x[idx+3] 
        y = self.tr_y[idx:idx+26,0]
        return x, y

@register_model
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

    def __len__(self):
        return len(self.tr_x) - 4

    def __getitem__(self, idx):
        x = self.tr_x[idx:idx+4, :, :, :, 0]
        x = rearrange(x, 'a b c d -> b c d a')

        y = np.squeeze(self.tr_y[idx, :])
        return x, y

@register_model
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
        tdim, _, _, _ = tr_x.shape

        sstData_label = nc.Dataset(SSTFile_label)
        tr_y = sstData_label[labelName][:, :, 0, 0]

        tr_y_c = np.zeros((tdim,12))
        for i in range(tdim):
            mod = i%12
            tr_y_c[i,mod] = 1
        self.y_c = np.array(tr_y_c)


        self.tr_x = np.array(tr_x)
        self.tr_y = np.array(tr_y[:, :])

    def _batchsize(self):
        return self.tr_x.shape

    def __len__(self):
        return len(self.tr_x)

    def __getitem__(self, idx):
        return self.tr_x[idx], self.tr_y[idx, :]
        # , self.y_c[idx, :]

@register_model
class rddataset(D.Dataset):
    def __init__(self, SSTFile, SSTFile_label, sstName, hcName, labelName):
        sstData =  nc.Dataset(SSTFile)
        sst = sstData[sstName][:, 0, :, :] # 35532, 24, 72
        sst = rearrange(sst, 'a b c -> 1 a c b')

        hc = sstData[hcName][:, 0, :, :]
        hc = rearrange(hc, 'a b c -> 1 a c b')
        tr_x = np.append(sst, hc, axis = 0)
        del sst, hc

        tr_x = rearrange(tr_x, 'c n h w -> n w h c') #(2, 35532, 24, 72) -> (35532, 72, 24, 2)

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

@register_model
class oisst2(D.Dataset):
    def __init__(self, sst_fp, hc_fp, mode, input_month = 3):
        if mode == 'train':
            endoflist = 408 - input_month
        elif mode == 'valid':
            endoflist = (444 - 408) - input_month
        
        sstData =  nc.Dataset(Path(sst_fp))['ssta'][:,::-1]
        hcData =  nc.Dataset(Path(hc_fp))['hca'][:,::-1]

        sst = pd.DataFrame(rearrange(sstData, 'a b c -> a (b c)'))
        sst = sst.fillna(0)
        sst = rearrange(sst.to_numpy(), 'a (w h) -> a w h', w = 360, h = 180)

        hc = pd.DataFrame(rearrange(hcData, 'a b c -> a (b c)'))
        hc = hc.fillna(0)
        hc = rearrange(hc.to_numpy(), 'a (w h) -> a w h', w = 360, h = 180)
        hc = (hc - hc.max()) / (hc.max() - hc.min())

        sst = rearrange(sst, 'a b c -> 1 a b c')
        sst = self.make_n_monthdata(sst, input_month)
        sst = sst[:,-endoflist:,:,:]    # sst = rearrange(sst[:,-endoflist:,:,:], 'a b c d -> 1 b a c d')


        hc = rearrange(hc, 'a b c -> 1 a b c')
        # hc = np.stack((hc[0,:-2,:,:], hc[0,1:-1,:,:], hc[0,2:,:,:]), axis=0)
        hc = self.make_n_monthdata(hc, input_month)
        # hc = rearrange(hc[:,-endoflist:,:,:], 'a b c d -> 1 b a c d')
        hc = hc[:,-endoflist:,:,:]

        # hc = np.expand_dims(hc[-endoflist:,:,:], axis = 0) #1, endoflist, 180, 360
        self.tr_x = np.append(sst, hc, axis = 0) # 6, 456, 180, 360
        self.tr_x = np.array(rearrange(self.tr_x, 'c b h w -> b c h w'), dtype=np.float32)

        self.tr_y = np.array(np.mean(np.mean(sstData[:,80:90,190:258], axis=-1), axis=-1)[-endoflist:], dtype=np.float32)

        del sst, hc

    def make_n_monthdata(self, x, n):
        tmp = []
        for i in range(n):
            tmp.append(x[0, i:-(n-i)])
        return np.stack(tmp, axis=0)

    def __len__(self):
        return len(self.tr_x) - 23

    def __getitem__(self, idx):
        x = self.tr_x[idx] 
        y = self.tr_y[idx:idx+23]
        return x, y

@register_model
class oisst3(D.Dataset):
    def __init__(self, sst_fp, hc_fp, mode, input_month = 3):
        if mode == 'train':
            endoflist = 408 - input_month
        elif mode == 'valid':
            endoflist = 444 - input_month
        
        sstData =  nc.Dataset(Path(sst_fp))['ssta'][:,::-1]
        hcData =  nc.Dataset(Path(hc_fp))['hca'][:,::-1]

        sst = pd.DataFrame(rearrange(sstData, 'a b c -> a (b c)'))
        sst = sst.fillna(0)
        sst = rearrange(sst.to_numpy(), 'a (w h) -> a w h', w = 360, h = 180)
        hc = pd.DataFrame(rearrange(hcData, 'a b c -> a (b c)'))
        hc = hc.fillna(0)
        hc = rearrange(hc.to_numpy(), 'a (w h) -> a w h', w = 360, h = 180)
        hc = (hc - hc.max()) / (hc.max() - hc.min())



        sst = rearrange(sst, 'a b c -> 1 a b c')
        # sst = np.stack((sst[0,:-2,:,:], sst[0,1:-1,:,:], sst[0,2:,:,:]), axis=0)
        sst = self.make_n_monthdata(sst, input_month)
        sst = rearrange(sst[:,-endoflist:,:,:], 'a b c d -> 1 b a c d')
        # sst = np.expand_dims(sst[-endoflist:,:,:], axis = 0) #1, endoflist, 180, 360

        hc = rearrange(hc, 'a b c -> 1 a b c')
        # hc = np.stack((hc[0,:-2,:,:], hc[0,1:-1,:,:], hc[0,2:,:,:]), axis=0)
        hc = self.make_n_monthdata(hc, input_month)
        hc = rearrange(hc[:,-endoflist:,:,:], 'a b c d -> 1 b a c d')

        # hc = np.expand_dims(hc[-endoflist:,:,:], axis = 0) #1, endoflist, 180, 360
        self.tr_x = np.append(sst, hc, axis = 0) # 2, 405, 3, 180, 360
        self.tr_x = np.array(rearrange(self.tr_x, 'c b d h w -> b c w h d'), dtype=np.float32)

        self.tr_y = np.array(np.mean(np.mean(sstData[:,80:90,190:258], axis=-1), axis=-1)[-endoflist:], dtype=np.float32)

    def make_n_monthdata(self, x, n):
        tmp = []
        for i in range(n):
            tmp.append(x[0, i:-(n-i)])
        return np.stack(tmp, axis=0)

    def __len__(self):
        return len(self.tr_x) - 23

    def __getitem__(self, idx):
        x = self.tr_x[idx] 
        y = self.tr_y[idx:idx+23]
        return x, y