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

from lstf.model.mdl import RFB_Transformer
from lstf.datasets.basicdatasets import basicdataset
from lstf.metric import CorrelationSkill
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
        "gpu": 0,
    })

    # Directories
    # Dataset for pretrain
    Folder = fp
    dataFolder = "./local/Dataset/Ham" #"./""./"

    SSTFile_val = dataFolder+'/godas.input.1980_2017.nc'
    SSTFile_val_label = dataFolder+'/godas.label.1980_2017.nc'

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
    valset = basicdataset(SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')
    batch_size = len(valset) // 1                             # batch size
    reg = lengthscale**2 * (1 - dr) / (2. * batch_size * tau) # L2 regularization weight for Bayesian learning
    testloader = DataLoader(valset, batch_size = batch_size, shuffle=False)

    test_step = len(testloader)

    assemble_real_nino = np.zeros((len(valset), 23))
    assemble_pred_nino = np.zeros((len(valset), 23))

    model = RFB_Transformer(in_channels=2, out_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=regularizer_rate, betas=(0.9, 0.999))
    model.load_state_dict(torch.load(f'{Folder}{dd}/{dd}.pth', map_location=device))
    model.eval()
    
    bayesianIter = 1

    with torch.no_grad() :
        for i, (batch, ansnino) in enumerate(testloader):
            batch = torch.tensor(batch, dtype=torch.float32).to(device=device)
            ansnino = torch.tensor(ansnino, dtype=torch.float32).to(device=device)
            idx = batch.shape[0]*i
            uncertaintyarry_nino = np.zeros((bayesianIter, batch_size, 23))
            for b in range(int(bayesianIter)):
                output = model(batch) # inference
                prednino = output.detach().cpu().numpy()
                uncertaintyarry_nino[b, :, :] = prednino

            assemble_real_nino[idx:idx+batch_size, :] = ansnino.cpu().numpy()

            assemble_pred_nino[idx:idx+batch_size, :] += np.mean(uncertaintyarry_nino, axis=0)

            del batch
            del ansnino

    mse = mean_squared_error(assemble_pred_nino, assemble_real_nino, multioutput='raw_values')
    print(mse)

    exit()
    
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

    np.savetxt(f'{Folder}/eval_6/correlation.csv',corr,delimiter=",")

    # print(assemble_pred_nino)
    np.save(f"{Folder}/eval_6/lead_assemble_real_nino", assemble_real_nino) # 길이가 valset인 것이 ensemble 갯수 만큼 들어있음
    np.save(f"{Folder}/eval_6/lead_assemble_pred_nino", assemble_pred_nino)

def compare():
    baseline = [0.90375662, 0.89310532, 0.85104236, 0.80583616, 0.74700119, 0.68805024
                ,0.64004255, 0.61192831, 0.58498279, 0.56291843, 0.53021225, 0.51725378
                ,0.48547276, 0.47183246, 0.44534206, 0.43026313, 0.40954246, 0.39029018
                ,0.38272185, 0.37395304, 0.36700487, 0.35023261, 0.32886644]

    baseline_mse = [0.14245229 ,0.15176659 ,0.21113602 ,0.2803528  ,0.35378224 ,0.42568714
                    ,0.48938734 ,0.53120054 ,0.56884297 ,0.59894049 ,0.62406066 ,0.64691595
                    ,0.665724   ,0.685555   ,0.70628766 ,0.71863057 ,0.73469082 ,0.74410355
                    ,0.75489915 ,0.75465311 ,0.75547143 ,0.75034072 ,0.74497082]

    trans = [9.37E-01
                ,9.29E-01
                ,9.00E-01
                ,8.67E-01
                ,8.30E-01
                ,7.85E-01
                ,7.40E-01
                ,7.02E-01
                ,6.67E-01
                ,6.36E-01
                ,6.01E-01
                ,5.65E-01
                ,5.33E-01
                ,5.10E-01
                ,4.89E-01
                ,4.72E-01
                ,4.51E-01
                ,4.34E-01
                ,4.12E-01
                ,3.95E-01
                ,3.76E-01
                ,3.60E-01
                ,3.39E-01
                ]

    trans_mse = [0.15202061 ,0.16552944 ,0.23341914 ,0.29851638 ,0.36664409 ,0.43075233
                ,0.47981692 ,0.50226231 ,0.52887196 ,0.55165997 ,0.58045937 ,0.60744801
                ,0.63956582 ,0.65761316 ,0.68639857 ,0.70053593 ,0.71932677 ,0.72996341
                ,0.74831092 ,0.75401899 ,0.75376829 ,0.75138524 ,0.76215806]

    timeline = np.arange(0, 23)

    plt.plot(timeline, np.sqrt(baseline_mse), marker='', color='blue', linewidth=1, label="baseline")
    plt.plot(timeline, np.sqrt(trans_mse), marker='', color='red', linewidth=1, linestyle='dashed', label="transformer")
    plt.legend()
    plt.savefig('mse_compare.png', orientation='landscape', bbox_inches='tight')
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
    # plotresult("./local/3DRFB_Transformer_encoders/", '5')
    compare()
    # check_last_model()
