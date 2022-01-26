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


def getcorr():
    # twodimcnn = [0.9294535,0.91590372,0.87245369,0.81624529,0.76330807,0.71168656
    #         ,0.66971695,0.6255489,0.59937599,0.59152533,0.56929781,0.57241524
    #         ,0.56720474,0.53959906,0.53506053,0.51028036,0.48267573,0.44623802
    #         ,0.42490706,0.39693028,0.35924735,0.32473416,0.28993454]

    twodimcnn = [0.87797048,0.87023331,0.83614309,0.80246051,0.76957703,0.73951732
                ,0.71618847,0.69847897,0.68389941,0.66393678,0.64016199,0.61270817
                ,0.58159395,0.5464659,0.51856569,0.49069667,0.45057754,0.40187238
                ,0.33761924,0.27528526,0.23281086,0.1798104,0.16641316]

    baseline = [0.90375662, 0.89310532, 0.85104236, 0.80583616, 0.74700119, 0.68805024
            ,0.64004255, 0.61192831, 0.58498279, 0.56291843, 0.53021225, 0.51725378
            ,0.48547276, 0.47183246, 0.44534206, 0.43026313, 0.40954246, 0.39029018
            ,0.38272185, 0.37395304, 0.36700487, 0.35023261, 0.32886644]

    trans = [9.37E-01,9.29E-01,9.00E-01,8.67E-01,8.30E-01,7.85E-01,7.40E-01
            ,7.02E-01,6.67E-01,6.36E-01,6.01E-01,5.65E-01,5.33E-01,5.10E-01
            ,4.89E-01,4.72E-01,4.51E-01,4.34E-01,4.12E-01,3.95E-01,3.76E-01
            ,3.60E-01,3.39E-01]

    twoloss = [0.94272792 ,0.92808749 ,0.88417982 ,0.83303625, 0.77855154, 0.71762297
    ,0.66378813 ,0.62712898 ,0.59591068 ,0.57344433, 0.54992418, 0.51244391
    ,0.49698463 ,0.4959474  ,0.4946409  ,0.49071608, 0.47689809, 0.45671596
    ,0.43125589 ,0.42269416 ,0.40528805 ,0.38918046, 0.37052003]

    corrloss = [0.78085976 ,0.76601635 ,0.70474282, 0.65893893, 0.59845962, 0.56696576
                ,0.50374322 ,0.47880643 ,0.46453552, 0.45799808, 0.41300029, 0.40151784
                ,0.39061753 ,0.37610801 ,0.29728042, 0.23347832, 0.19291504, 0.18244784
                ,0.15851624 ,0.10160118 ,0.11149167, 0.11137596, 0.11123336]


    return baseline, trans, twodimcnn, twoloss, corrloss

def getmse():
    baseline_mse = [0.14245229 ,0.15176659 ,0.21113602 ,0.2803528  ,0.35378224 ,0.42568714
                ,0.48938734 ,0.53120054 ,0.56884297 ,0.59894049 ,0.62406066 ,0.64691595
                ,0.665724   ,0.685555   ,0.70628766 ,0.71863057 ,0.73469082 ,0.74410355
                ,0.75489915 ,0.75465311 ,0.75547143 ,0.75034072 ,0.74497082]



    trans_mse = [0.15202061 ,0.16552944 ,0.23341914 ,0.29851638 ,0.36664409 ,0.43075233
                ,0.47981692 ,0.50226231 ,0.52887196 ,0.55165997 ,0.58045937 ,0.60744801
                ,0.63956582 ,0.65761316 ,0.68639857 ,0.70053593 ,0.71932677 ,0.72996341
                ,0.74831092 ,0.75401899 ,0.75376829 ,0.75138524 ,0.76215806]

    # twodimcnn_mse = [0.10440155,0.12305518,0.18533027,0.26773953,0.33833587,0.40438726
    #                 ,0.45247577,0.50099879,0.52999656,0.54508839,0.57087975,0.57319739
    #                 ,0.58352541,0.60487012,0.60630755,0.62621512,0.6505984,0.68400098
    #                 ,0.70633535,0.73157495,0.75466525,0.77619356,0.79621094]

    twodimcnn_mse = [0.26267617,0.20164667,0.25757204,0.32053927,0.39002035,0.43877803
                    ,0.45900149,0.47882055,0.49832767,0.52270331,0.54337768,0.57269402
                    ,0.60194564,0.63117916,0.64668539,0.66478106,0.70156517,0.73569109
                    ,0.77526975,0.80955846,0.82480109,0.84073801,0.84677622]


    twoloss_mse = [0.1038604  ,0.12485667 ,0.19513106 ,0.29182439, 0.36944173, 0.43874408
    ,0.48725133 ,0.51639513 ,0.54440538 ,0.56261593, 0.58601967, 0.62211701
    ,0.63189618 ,0.62910985 ,0.63067247 ,0.63401239, 0.64879604, 0.67467751
    ,0.71918714 ,0.71807392 ,0.73776863 ,0.7480545,  0.76219895]



    return baseline_mse, trans_mse, twodimcnn_mse, twoloss_mse


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
