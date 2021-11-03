from matplotlib.pyplot import axis
import numpy as np
# import tensorflow as tf
from netCDF4 import Dataset
from scipy.io import netcdf
from scipy.stats.stats import pearsonr
import math
import collections

np.random.seed(722)

def Compute3MonthAvgAnomaly(sst, time, longitude, latitude, dur):
    # offset = int(dur/2)
    subSST = sst[time:time+dur, longitude[0]:longitude[1], latitude[0]:latitude[1]]
    anomaly = np.nanmean(subSST)
    del subSST
    return anomaly

def GetBatch_All(sst, hc, indices, timeSeq):
    sstSet = []
    hcSet = []
    indices = indices - timeSeq + 1
    month = [(x+1) %12 for x in range(indices, indices + 3)]
    for t in range(timeSeq):
        mSST = sst[indices, :, :]
        sstSet.append(mSST)
        mHC = hc[indices, :, :]
        hcSet.append(mHC)
        indices = indices + 1
    condition = getcondition(sst.shape, collections.deque(month))
    batchSST = np.stack(sstSet, axis=2)
    batchSST = np.concatenate([batchSST, condition], axis = 2)
    batchHC = np.stack(hcSet, axis=2)
    batchHC = np.concatenate([batchHC, condition], axis = 2)
    del sstSet
    del hcSet
    return batchSST, batchHC
def getcondition(shape, months) : 
    array = np.zeros((shape[1], shape[2], 12))
    while months : 
        month = months.popleft()
        if (month) % 12 == 1 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 2 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 3 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 4 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 5 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 6 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 7 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 8 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 9 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 10 : 
            array[:, :, month-1] = 1
        if (month) % 12 == 11: 
            array[:, :, month-1] = 1
        if (month) % 12 == 12 : 
            array[:, :, month-1] = 1
    return array

def CorrelationSkill(real, pred):
    size = real.shape[0] // 12
    real = np.reshape(real, (size, 12))
    pred = np.reshape(pred, (size, 12))
    corrAvg = 0
    for i in range(12):
        corr = np.corrcoef(real[:, i], pred[:, i])[0][1]
        corrAvg += corr
    corrAvg = corrAvg / 12.0
    return corrAvg

def Hitrate(real, pred):
    size = real.shape[0] // 12
    real = np.reshape(real, (size, 12))
    pred = np.reshape(pred, (size, 12))
    array_hitrate = [] 
    for i in range(12):
        hitrate = 0 
        for j in range(size) : 
            if real[j, i] == pred[j, i] :
                hitrate += 1
        hitrate /= size
        array_hitrate.append(hitrate)
    return array_hitrate 

def CalculateAnomaly(sst):
    sstMeanMaps = []
    for i in range(12):
        indices = np.arange(i, sst.shape[0], 12)
        print(indices[-2:])
        sstSub = sst[indices, :, :]
        sstMeanMap = np.mean(sstSub, axis=0)
        sstMeanMaps.append(sstMeanMap)

    for i in range(sst.shape[0]):
        sstMeanMap = sstMeanMaps[i%12]
        sst[i, :, :] = sst[i, :, :] - sstMeanMap
    return sst

def ReadData(filename):
    data = Dataset(filename, 'r', format = 'NETCDF4', )
    return data
