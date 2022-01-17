#!/usr/bin/env python
#JHK
import numpy as np
from netCDF4 import Dataset
from sklearn.metrics import accuracy_score


def CorrelationSkill(real, pred):
    size = int(len(real)/12)
    zeros_real = np.zeros(size * 12)
    zeros_pred = np.zeros(size * 12)
    try :
        for i in range(size * 12) :
            zeros_pred[i] = pred[i]
            zeros_real[i] = real[i]
    except :
        pass

    real = np.reshape(zeros_real, (size, 12))
    pred = np.reshape(zeros_pred, (size, 12))
    real = np.float32(real)
    pred = np.float32(pred)
    real = np.nan_to_num(real, 0.0)
    pred = np.nan_to_num(pred, 0.0)
    corrAvg = 0
    for i in range(12):
        corr = np.corrcoef(real[:, i], pred[:, i])[0][1]
        corrAvg += corr
    corrAvg = corrAvg / 12.0
    return corrAvg

nino34 = open('./v01_tr40/C35D50/nino34.gdat', 'r')
nino34 = np.fromfile(nino34, np.float32)
nino34 = nino34.reshape(-1, 23)

f = Dataset('./Data_validation/godas.label.1980_2017.nc','r')
test_y = f.variables['pr'][:,:,0,0]
f.close()

print(nino34.shape, test_y.shape)

corr = np.zeros(23)
for i in range(23):
    corr[i] = CorrelationSkill(nino34[:, i], test_y[:, i])

print(corr)
np.savetxt('2Doutput_correlation.csv', corr, delimiter=",")

# classification
month = open('./v01_tr40/C35D50/month.gdat', 'r')
month = np.fromfile(month, np.float32)
month = month.reshape(-1, 12)
print(month.shape)

month_class_pred = np.argmax(month, axis=1)
print(month_class_pred)

month_gt = np.zeros(month.shape[0])
for i in range(month.shape[0]):
    month_gt[i] = i%12

print(month_gt)
accuracy = np.zeros(1)
accuracy[0] = accuracy_score(month_gt, month_class_pred)
print('Accuracy = {}'.format(accuracy))
np.savetxt('2Doutput_accuracy.csv',accuracy,delimiter=",")