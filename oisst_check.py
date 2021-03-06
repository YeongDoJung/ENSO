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
from ltsf.datasets import dataset 
from ltsf.metric import CorrelationSkill
import easydict
import argparse
import csv
import pandas as pd

from sklearn.metrics import accuracy_score, mean_squared_error

def plotresult(fp):
    # np.random.seed(0)
    # random.seed(0)
    # torch.manual_seed(0)

    # Arguments
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--data_inputmonth', type=int, default=3)
    parser.add_argument('--data_targetmonth', type=int, default=24)

    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    a = fp.split('/')
    dd = str(a[-1])[0:-4] #eval_nnn.pth

    Folder = a[0] + '/' + a[1]
    dataFolder = "./local/Dataset/oisst" #"./""./"

    SSTFile_val = dataFolder+'/test/sst.nc'
    HCFile_val = dataFolder+'/test/hc.nc'

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # eg) local\oisst_trf_mse\eval_188\eval_188.pth

    # Dataset for training
    valset =  dataset.__dict__[args.dataset](SSTFile_val, HCFile_val, 'valid', input_month = args.data_inputmonth, target_month = args.data_targetmonth)
    batch_size = 1 # len(valset) // 1                             # batch size
    testloader = DataLoader(valset, batch_size = batch_size, shuffle=False)

    assemble_real_nino = np.zeros((len(valset), args.data_targetmonth))
    assemble_pred_nino = np.zeros((len(valset), args.data_targetmonth))

    model = build.__dict__[args.model](num_classes = args.data_targetmonth).to(device=device)
    model.load_state_dict(torch.load(f'{Folder}/{dd}/{dd}.pth', map_location=device))
    model.eval()
    
    bayesianIter = 1

    with torch.no_grad() :
        for i, (batch,ansnino, idx) in enumerate(testloader):
            batch = torch.tensor(batch, dtype=torch.float32).to(device=device)
            ansnino = torch.tensor(ansnino, dtype=torch.float32).to(device=device)
            idx = batch.shape[0]*i
            uncertaintyarry_nino = np.zeros((bayesianIter, batch_size, args.data_targetmonth))
            for b in range(int(bayesianIter)):
                output = model(batch) # inference
                prednino = output[:,:].detach().cpu().numpy()
                uncertaintyarry_nino[b, :, :] = prednino

            assemble_real_nino[idx:idx+batch_size, :] = ansnino[:,:].cpu().numpy()
            assemble_pred_nino[idx:idx+batch.shape[0], :] += np.mean(uncertaintyarry_nino, axis=0)


    mse = mean_squared_error(assemble_pred_nino, assemble_real_nino, multioutput='raw_values')
    print(mse)

    corr = np.zeros(args.data_targetmonth)
    for i in range(args.data_targetmonth):
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
        plt.savefig(Folder + f"/{dd}/NinoPred_" + str(i).zfill(6) + ".png", orientation='landscape', bbox_inches='tight')
        plt.show()
        plt.close()

    print(corr)

    np.savetxt(f'{Folder}/{dd}/correlation.csv',corr,delimiter=",")

    # print(assemble_pred_nino)
    np.save(f"{Folder}/{dd}/lead_assemble_real_nino", assemble_real_nino) # ????????? valset??? ?????? ensemble ?????? ?????? ????????????
    np.save(f"{Folder}/{dd}/lead_assemble_pred_nino", assemble_pred_nino)

    return mse, corr

def compare(mse, *corrs, fp, num, label):
    aa = fp + 'eval_' + num
    timeline = np.linspace(0, 22, 1)

    # plt.plot(timeline, np.sqrt(baseline_mse), marker='', color='blue', linewidth=1, label="baseline")
    # plt.plot(timeline, np.sqrt(trans_mse), marker='', color='red', linewidth=1, label="rfb_transformer")
    # # plt.plot(timeline, np.sqrt(twolossmse), marker='', color='green', linewidth=1, label="vit_twoloss")
    # plt.plot(timeline, np.sqrt(mse), marker='', color='purple', linewidth=1, label=label)
    # plt.legend()
 
    # plt.savefig(aa + '/mse_compare.png', orientation='landscape', bbox_inches='tight')
    # plt.show()
    # plt.close()

    # plt.cla()

    axes = plt.axes()
    axes.set_xlim([0, 23])
    axes.set_ylim([0,1])

    for corr in corrs:
        plt.plot(timeline, corr, marker='', linewidth=1, label=label)

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
    # name & fp
    oisst_trf_fp = {'oisst_transformer_mse' : 'local/oisst_trf_mse/eval_188/eval_188.pth',
                    'oisst_transformer_FrechetGELV' : 'local/oisst_trf_frechet/eval_10/eval_10.pth',
                    'oisst_transformer_GumbelGELV' :'local/oisst_trf_gumbel/eval_16/eval_16.pth',
                    'oisst_transformer_WeightedMSE' : 'local/oisst_trf_weightedmse/eval_19/eval_19.pth'}

    oisst_lstm_fp = {'oisst_lstm_mse' : 'local/oisst_lstm_rfb4_rmse/eval_79/eval_79.pth',
                    'oisst_lstm_FrechetGELV' : 'local/oisst_lstm_rfb4_frechet/eval_82/eval_82.pth',
                    'oisst_lstm_GumbelGELV' :'local/oisst_lstm_rfb4_gumbel/eval_122/eval_122.pth',
                    'oisst_lstm_WeightedMSE' : 'local/oisst_lstm_rfb4_weightedrmse/eval_559/eval_559.pth'}

    oisst_pvt_fp = {'oisst_pvt_mse' : 'local/oisst_pvt/eval_19/eval_19.pth',
                    'oisst_pvt_FrechetGELV' : 'local/oisst_pvt_frechet_/eval_17/eval_17.pth',
                    'oisst_pvt_GumbelGELV' :'local/oisst_pvt_gumbel_/eval_14/eval_14.pth'}
                    # 'oisst_pvt_WeightedMSE' : 'local/sep_pvt_weightedrmse/eval_559/eval_559.pth'}

    sep_pvt_fp = {'sep_pvt_mse' : 'local/sep_pvt_mse_/eval_44/eval_44.pth',
                    'sep_pvt_FrechetGELV' : 'local/sep_pvt_fr/eval_968/eval_968.pth',
                    'sep_pvt_GumbelGELV' :'local/sep_pvt_gv/eval_1554/eval_1554.pth'}

    onemonth = {'rmse' : 'local/1m_predict_sep_pvt_/eval_77/eval_77.pth',
                'frechet' : 'local/1m_predict_sep_pvt_fr/eval_70/eval_70.pth',
                'gumbel': 'local/1m_predict_sep_pvt_gv/eval_275/eval_275.pth'}

    year = {'rmse' : 'local/12m_predict_sep_pvt_/eval_179/eval_179.pth',
                'frechet' : 'local/12m_predict_sep_pvt_fr/eval_19/eval_19.pth',
                'gumbel': 'local/12m_predict_sep_pvt_gv/eval_80/eval_80.pth'}

    tmp = {'rmse': 'local/st_enc_mse/eval_194/eval_194.pth',
            'frechet': 'local/st_enc_fr/eval_86/eval_86.pth',
            'gumbel' : 'local/st_enc_gb/eval_196/eval_196.pth'}



    mses, corrs = [], []
    
    for i in year:
        mse, corr = plotresult(year[i])
        tmp[i] = corr

    # plt.plot([0.5]*23, marker='r--')
    for i in tmp:
        plt.plot(tmp[i], marker='', linewidth=1, label=i)
    plt.legend()
    plt.savefig('cc.png', orientation='landscape', bbox_inches='tight')

    plt.show()
    plt.close()


    # compare(mse, corr, fp, num, label = 'teacherforcing_sep_vit')
    # check_last_model()
