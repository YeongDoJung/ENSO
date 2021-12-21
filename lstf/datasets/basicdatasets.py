import numpy as np
import netCDF4 as nc

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
        return self.tr_x[idx], self.tr_y[idx, :]