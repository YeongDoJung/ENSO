import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import math
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ltsf.model import build
from ltsf.datasets.dataset import basicdataset, tdimdataset, tfdataset, tgtdataset, oisst2
from ltsf.metric import CorrelationSkill
import easydict
import csv
import pandas as pd

from sklearn.metrics import accuracy_score, mean_squared_error


def plotresult(fp, num):


    # np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)

    # Arguments
    args = easydict.EasyDict({
        "gpu": 3,
    })

    # Directories
    # Dataset for pretrain
    Folder = fp
    dataFolder = "./local/Dataset/oisst" #"./""./"

    SSTFile_val = dataFolder+'/test/sst.nc'
    HCFile_val = dataFolder+'/test/hc.nc'

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # Set Hyper-parameters
    regularizer_rate = 0.00001  #L2 regularization
    numEpoch =  100              # No. Epoch
    learning_rate = 0.0001      # Initial Learning Rate
    n_cycles = 4                # No. cycles in Cosine Annealing
    epochs_per_cycle = math.floor(numEpoch / n_cycles)  # No. epochs for each cycle

    dr = 0.0                   # Dropout rate for Bayesian learning
    tau = 1.0                   # Weight for the batch size in regularization weight calculation (Bayesian learning)
    lengthscale = 1e-2          # Default regularization weight (L2)
    noF = 16                    # Initial No. filters
    num_layer = 256             # Feature size of 1st fully-connected layer
    num_answer = 2              # No. answers(3=3.4/ep/cp)

    dd = 'eval_' + num

    # Dataset for training
    valset = oisst2(SSTFile_val, HCFile_val, 'valid')
    batch_size = len(valset) // 1                             # batch size
    reg = lengthscale**2 * (1 - dr) / (2. * batch_size * tau) # L2 regularization weight for Bayesian learning
    testloader = DataLoader(valset, batch_size = batch_size, shuffle=False)

    test_step = len(testloader)

    assemble_real_nino = np.zeros((len(valset), 23))
    assemble_pred_nino = np.zeros((len(valset), 23))

    model = build.pyramid().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=regularizer_rate, betas=(0.9, 0.999))
    model.load_state_dict(torch.load(f'{Folder}{dd}/{dd}.pth', map_location=device))
    model.eval()
    
    bayesianIter = 1

    with torch.no_grad() :
        for i, (batch,ansnino) in enumerate(testloader):
            batch = torch.tensor(batch, dtype=torch.float32).to(device=device)
            ansnino = torch.tensor(ansnino, dtype=torch.float32).to(device=device)
            idx = batch.shape[0]*i
            uncertaintyarry_nino = np.zeros((bayesianIter, batch_size, 23))
            for b in range(int(bayesianIter)):
                output = model(batch) # inference
                prednino = output[:,-23:].detach().cpu().numpy()
                uncertaintyarry_nino[b, :, :] = prednino

            assemble_real_nino[idx:idx+batch_size, :] = ansnino[:,-23:].cpu().numpy()
            assemble_pred_nino[idx:idx+batch.shape[0], :] += np.mean(uncertaintyarry_nino, axis=0)


    mse = mean_squared_error(assemble_pred_nino, assemble_real_nino, multioutput='raw_values')
    print(mse)

    corr = np.zeros(23)
    for i in range(23):
        corr[i] = CorrelationSkill(assemble_real_nino[:, i], assemble_pred_nino[:, i])
        print('Save prediction: lead = {}'.format(i) )
        inputTimeSeq = assemble_real_nino.shape[0]
        dwidth = 800
        dpi = 90
        dheight = 180
        plt.figure(figsize=(dwidth/dpi, dheight/dpi))
        timeValues = np.arange(0, inputTimeSeq)
        plt.plot(timeValues, assemble_real_nino[:, i], marker='', color='blue', linewidth=1, label="Measurement")
        plt.plot(timeValues, assemble_pred_nino[:, i], marker='', color='red', linewidth=1, linestyle='dashed', label="Prediction")
        plt.savefig(Folder + f"{dd}/NinoPred_" + str(i).zfill(6) + ".png", orientation='landscape', bbox_inches='tight')
        plt.show()
        plt.close()

    print(corr)

    np.savetxt(f'{Folder}{dd}/correlation.csv',corr,delimiter=",")

    # print(assemble_pred_nino)
    np.save(f"{Folder}{dd}/lead_assemble_real_nino", assemble_real_nino) # 길이가 valset인 것이 ensemble 갯수 만큼 들어있음
    np.save(f"{Folder}{dd}/lead_assemble_pred_nino", assemble_pred_nino)

    return mse, corr

def compare(mse, corr, fp, num, label):
    aa = fp + 'eval_' + num

    baseline_mse, trans_mse, twodimmse, twolossmse = getmse()
    baseline, trans, twodim, twoloss, corrloss = getcorr()

    timeline = np.linspace(0, 22, 1)

    axes = plt.axes()
    axes.set_ylim([0,1])
    plt.plot(timeline, np.sqrt(baseline_mse), marker='', color='blue', linewidth=1, label="baseline")
    plt.plot(timeline, np.sqrt(trans_mse), marker='', color='red', linewidth=1, label="rfb_transformer")
    # plt.plot(timeline, np.sqrt(twolossmse), marker='', color='green', linewidth=1, label="vit_twoloss")
    plt.plot(timeline, np.sqrt(mse), marker='', color='purple', linewidth=1, label=label)
    plt.legend()
 
    plt.savefig(aa + '/mse_compare.png', orientation='landscape', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.cla()


    axes = plt.axes()
    axes.set_xlim([0, 23])
    axes.set_ylim([0,1])
    plt.plot(timeline, baseline, marker='', color='blue', linewidth=1, label="baseline")
    plt.plot(timeline, trans, marker='', color='red', linewidth=1, label="rfb_transformer")
    # plt.plot(timeline, np.sqrt(twoloss), marker='', color='green', linewidth=1, label="vit_twoloss")
    plt.plot(timeline, corr, marker='', color='purple', linewidth=1, label=label)
    plt.legend()

    plt.savefig(aa + '/corr_compare.png', orientation='landscape', bbox_inches='tight')
    plt.show()
    plt.close()

def check_last_model():
    sp = './local/3DRFB_Transformer_decoder/'
    sps = os.listdir(sp)
    kk = []
    for i in sps:
        if os.path.isfile(sp+i+'/'+i+'.pth'):

            print(i)
            kk.append(i)

    kk = sorted(kk)
    print(kk)

if __name__ == '__main__':
    wo_fp, wo_num = "local/oisst_pvt_1/", '693'
    fp, num = "local/oisst_pvt/", '265'
    wo_mse, wo_corr = plotresult(wo_fp, wo_num)
    mse, corr = plotresult(fp, num)

    plt.plot(corr, marker='', color='blue', linewidth=1, label="TeacherForcing")
    plt.plot(wo_corr, marker='', color='red', linewidth=1, label="w/o_TeacherForcing")
    plt.legend()
    plt.savefig('cc.png', orientation='landscape', bbox_inches='tight')

    plt.show()
    plt.close()


    # compare(mse, corr, fp, num, label = 'teacherforcing_sep_vit')
    # check_last_model()
