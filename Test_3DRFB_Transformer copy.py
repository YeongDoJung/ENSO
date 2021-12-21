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

from lstf.model.mdl_decoder import RFB_Transformer
from lstf.datasets.basicdatasets import basicdataset
from Parts import *
import easydict
import csv
import pandas as pd

from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Random seed 
    # torch.backends.cudnn.deterministic = True 
    # torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # Arguments
    args = easydict.EasyDict({
        "gpu": 1,
    })

    # Directories
    # Dataset for pretrain
    Folder = "./local/3DRFB_Transformer/"
    folders = sorted(os.listdir(Folder))
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

    # Dataset for training
    valset = basicdataset(SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')
    batch_size = len(valset) // 1                             # batch size
    reg = lengthscale**2 * (1 - dr) / (2. * batch_size * tau) # L2 regularization weight for Bayesian learning
    testloader = DataLoader(valset, batch_size = batch_size, shuffle=False)

    test_step = len(testloader)

    assemble_real_nino = np.zeros((len(valset), 23))
    assemble_pred_nino = np.zeros((len(valset), 23))

    model = RFB_Transformer(in_channels=2, out_channels=16).to(device)
    for i in folders:
        if os.path.isfile(f'{Folder}/{i}/{i}.pth'):
            model.load_state_dict(torch.load(f'{Folder}/{i}/{i}.pth'))
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=regularizer_rate, betas=(0.9, 0.999))
            model.eval()
            
            bayesianIter = 1

            with torch.no_grad() :
                for i, (batch, ansnino) in enumerate(testloader):
                    batch = torch.tensor(batch, dtype=torch.float32).to(device=device)
                    ansnino = torch.tensor(ansnino, dtype=torch.float32).to(device=device)
                    ansnino = 1 / (1 + torch.exp(-1*ansnino)) - 0.5
                    idx = batch.shape[0]*i
                    uncertaintyarry_nino = np.zeros((bayesianIter, batch_size, 23))
                    for b in range(int(bayesianIter)):
                        output = model(batch) # inference
                        output = 1 / (1 + torch.exp(-1*output))
                        prednino = output.detach().cpu().numpy()
                        uncertaintyarry_nino[b, :, :] = prednino

                    assemble_real_nino[idx:idx+batch_size, :] = ansnino.cpu().numpy()

                    assemble_pred_nino[idx:idx+batch_size, :] += np.mean(uncertaintyarry_nino, axis=0)

                    del batch
                    del ansnino
            
            corr = np.zeros(23)
            for i in range(23):
                corr[i] = dp.CorrelationSkill(assemble_real_nino[:, i], assemble_pred_nino[:, i])
                print(corr[i])
                print('Save prediction: lead = {}'.format(i) )
                inputTimeSeq = assemble_real_nino.shape[0]
                dwidth = 800
                dpi = 90
                dheight = 180
                plt.figure(figsize=(dwidth/dpi, dheight/dpi))
                timeValues = np.arange(0, inputTimeSeq)
                # plt.plot(timeValues, assemble_real_nino[:, i], marker='', color='blue', linewidth=1, label="Measurement")
                # plt.plot(timeValues, assemble_pred_nino[:, i], marker='', color='red', linewidth=1, linestyle='dashed', label="Prediction")
                plt.savefig(Folder + "/NinoPred_" + str(i).zfill(6) + ".png", orientation='landscape', bbox_inches='tight')
                # plt.show()
                # plt.close()
            
            np.savetxt(f'{Folder}/{i}/correlation.csv',corr,delimiter=",")

            # print(assemble_pred_nino)
            # np.save(f"{Folder}/{i}/lead_assemble_real_nino", assemble_real_nino) # 길이가 valset인 것이 ensemble 갯수 만큼 들어있음
            # np.save(f"{Folder}/{i}/lead_assemble_pred_nino", assemble_pred_nino)
        else:
            pass

