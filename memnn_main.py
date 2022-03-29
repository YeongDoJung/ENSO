from numpy.core.numeric import zeros_like
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import math
import numpy as np
import os
from pathlib import Path
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error

from ltsf import metric, util
from ltsf.datasets import dataset
from ltsf.model import build
import argparse
import tqdm

def pearson(pred, gt):
    allLoss = 0
    for i in range(pred.shape[0]):
        score = pred[i, :]
        target = gt[i, :]
        vx = score - torch.mean(score)
        vy = target - torch.mean(target)
        add = torch.sum((score - target) ** 2) / pred.shape[1]
        loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) 
        allLoss += 1.0 - loss + add*0.5
    allLoss /= pred.shape[0]
    return allLoss

def train(args, model, optimizer, trainset, criterion, writer):
    args = args
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.construct_memory(trainset)
    
    trainloader = tqdm.tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False), total=len(trainset)//args.batch_size)

    # Training
    model.train()

    trainloss = metric.AverageMeter()
    
    for i, (src, label) in enumerate(trainloader):
        src = src.clone().detach().requires_grad_(True).to(device=device)
        label = label.clone().detach().requires_grad_(True).to(device=device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True): 
            output = model(src)
            tl = criterion(output, label)
            trainloss.update(tl)

        scaler.scale(tl).backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.step(optimizer) 
        scaler.update()

        del src
        del label

        trainloader.set_description(f'{args.current_epoch}/{args.numEpoch} train loss = {trainloss.avg:.4f}-{tl:.4f}')

    # vis.add_data_point(title = 'loss/train', data = trainloss.avg, pos = args.current_epoch)

    writer.add_scalar('loss/train', trainloss.avg, args.current_epoch)

    trainloss.reset()




def valid(args, model, valset, criterion, writer):

    testloader = tqdm.tqdm(DataLoader(valset, batch_size=1, shuffle=False, drop_last=False), total=len(valset))
    valloss = metric.AverageMeter()
    model.eval()

    assemble_real_nino = np.zeros((len(valset), args.data_targetmonth))
    assemble_pred_nino = np.zeros((len(valset), args.data_targetmonth))

    with torch.no_grad() :
        for i, (src, label, tgt) in enumerate(testloader):
            src = src.clone().detach().requires_grad_(True).to(device=device)
            label = label.clone().detach().requires_grad_(True).to(device=device)

            idx = src.shape[0]*i
            uncertaintyarry_nino = np.zeros((1, src.shape[0], args.data_targetmonth))

            for b in range(int(1)):
                output = model(src)
                vl = val_crit(output, label)
                valloss.update(vl)
                uncertaintyarry_nino[b, :, :] = output[:,:].cpu()

                assemble_real_nino[idx:idx+src.shape[0], :] = label[:,:].cpu().numpy()

            assemble_pred_nino[idx:idx+src.shape[0], :] += np.mean(uncertaintyarry_nino, axis=0)
            
            del src
            del label

            testloader.set_description(f'{args.current_epoch}/{args.numEpoch} valid loss = {valloss.avg:.4f}-{vl:.4f}')

        # vis.add_data_point(title = 'loss/valid', data = valloss.avg, pos = args.current_epoch)
        
        corr = np.zeros(args.data_targetmonth)
        for i in range(args.data_targetmonth):
            corr[i] = metric.CorrelationSkill(assemble_real_nino[:, i], assemble_pred_nino[:, i])

        mse = mean_squared_error(assemble_pred_nino, assemble_real_nino)
        print(corr)

        util.ploter(corr, f'{Folder}/fig/{args.current_epoch}.png', args.data_targetmonth)

    if (valloss.avg) < args.valloss_best : 
        args.valloss_best = valloss.avg
        os.makedirs(f'{Folder}/eval_{args.current_epoch}/', exist_ok=True)
        torch.save(model.state_dict(), f'{Folder}/eval_{args.current_epoch}/eval_{args.current_epoch}.pth')
        np.savetxt(f'{Folder}/eval_{args.current_epoch}/eval_{args.current_epoch}_mse_{mse:.4f}_corr.csv', corr)
        print('[{}/{} , mean corr : {}'.format(args.current_epoch, args.numEpoch, np.mean(corr)))
        writer.add_scalar('corr/test', valloss.avg, args.current_epoch)
        writer.flush()
    writer.close()

    valloss.reset()
    args.current_epoch += 1

    return corr



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--numEpoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--name", type=str, default='res_enc_2')
    parser.add_argument('--data', type=int, default=0)
    parser.add_argument('--data_inputmonth', type=int, default=3)
    parser.add_argument('--data_targetmonth', type=int, default=24)

    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--crit', type=str, default='')
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument("--val_min", type=float, default=9999)
    parser.add_argument("--current_epoch", type=int, default=0)
    args = parser.parse_args()

    # vis = util.visualizer_visdom(env = args.name)

    GPU_NUM = args.gpu
    device = torch.device('cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # Set a Hyper-parameters
    # regularizer_rate = 0.0    #L2 regularization

    dr = 0.0                    # Dropout rate for Bayesian learning
    tau = 1.0                   # Weight for the batch size in regularization weight calculation (Bayesian learning)
    lengthscale = 0.1           # Default regularization weight (L2)
    reg = lengthscale**2 * (1 - dr) / (2. * args.batch_size * tau) # L2 regularization weight for Bayesian learning
    noF = 16                     # Initial No. filters

    num_layer =  256             # Feature size of 1st fully-connected layer
    num_answer = 2              # No. answers(3=3.4/ep/cp)

    minRMSE = 100.0             # minimum RMSE
    minUncertainty = 100.0      # minimum uncertainty

    leadMax = 24                # No. lead time

    args.valloss_best = 9999

    # Dataset for pretraining
    Folder = Path(str(Path(__file__).parent) + "/local/" + args.name)
    dataFolder = Path(str(Path(__file__).parent) + '/local/Dataset/') #"./"
    os.makedirs(f'{Folder}/fig/', exist_ok=True)

    # Dataset for training

    
    model = build.memorynn().to(device=device)    
    '''    
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        for submodule in model.modules():
            if str(submodule).startswith('Multi'):
                pass
            else:
                submodule.register_forward_hook(nan_hook)
    '''
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005, alpha=0.9)
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    lrsc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = 50)
    # criterion = nn.MSELoss(reduction='mean')
    criterion = metric.__dict__[args.crit]()
    val_crit = nn.MSELoss()

    corr_list = []
    
    torch.cuda.empty_cache()

    if not os.path.exists(f'{Folder}/'):
        os.makedirs(f'{Folder}/')
    writer = SummaryWriter(f'{Folder}/tensorboard/')

    for epoch in range(args.numEpoch):
        if args.data == 1:
            SSTFile_train_sst = dataFolder / 'oisst' / 'finetuning' / 'sst.nc'
            SSTFile_train_hc = dataFolder / 'oisst' / 'finetuning' / 'hc.nc'
            SSTFile_test_sst = dataFolder / 'oisst' / 'test' / 'sst.nc'
            SSTFile_test_hc = dataFolder / 'oisst' / 'test' / 'hc.nc'

            trainset = dataset.__dict__[args.dataset](SSTFile_train_sst, SSTFile_train_hc, 'train', input_month = args.data_inputmonth, target_month = args.data_targetmonth)
            valset = dataset.__dict__[args.dataset](SSTFile_test_sst, SSTFile_test_hc, 'valid', input_month = args.data_inputmonth, target_month = args.data_targetmonth)
        elif args.data == 2:
            SSTFile_train = dataFolder / 'Ham' / 'cmip5_tr.input.1861_2001.nc'
            SSTFile_train_label = dataFolder / 'Ham' / 'cmip5_tr.label.1861_2001_integrated.npy'
            SSTFile_val = dataFolder / 'Ham' / 'godas.input.1980_2017.nc'
            SSTFile_val_label = dataFolder / 'Ham' / 'godas.label.1980_2017_integrated.npy'

            #datasets_general_3D_alllead_add(SSTFile_train, SSTFile_train_label, SSTFile_train2, SSTFile_train_label2, lead, sstName='sst', hcName='t300', labelName='pr', noise = True) 
            trainset = dataset.__dict__[args.dataset](SSTFile_train, SSTFile_train_label, sstName='sst', hcName='t300', labelName='pr', currnet_epoch = args.current_epoch)  
            valset = dataset.__dict__[args.dataset](SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')
        elif args.data == 3:
            SSTFile_train = dataFolder / 'Ham' / 'cmip5_tr.input.1861_2001_mean.npy'
            SSTFile_train_label = dataFolder / 'Ham' / 'cmip5_tr.label.1861_2001_mean.npy'
            SSTFile_val = dataFolder / 'Ham' / 'godas.input.1980_2017.nc'
            SSTFile_val_label = dataFolder / 'Ham' / 'godas.label.1980_2017.nc'

            trainset = dataset.__dict__[args.dataset](SSTFile_train, SSTFile_train_label, sstName='sst', hcName='t300', labelName='pr') 
            valset = dataset.__dict__['basicdataset'](SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')

        elif args.data == 0:
            SSTFile_train = dataFolder / 'Ham' / 'cmip5_tr.input.1861_2001.nc'
            SSTFile_train_label = dataFolder / 'Ham' / 'cmip5_tr.label.1861_2001.nc'
            SSTFile_val = dataFolder / 'Ham' / 'godas.input.1980_2017.nc'
            SSTFile_val_label = dataFolder / 'Ham' / 'godas.label.1980_2017.nc'

            trainset = dataset.__dict__[args.dataset](SSTFile_train, SSTFile_train_label, sstName='sst', hcName='t300', labelName='pr') 
            valset = dataset.__dict__[args.dataset](SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train(args, model=model, optimizer=optimizer, trainset=trainset, criterion=criterion, writer = writer)
        c = valid(args, model=model, valset=valset, criterion=criterion, writer = writer)
        lrsc.step()
        corr_list.append(c)
        # test(args, model=model, testloader)

    with open(Folder.joinpath('corr.csv'), 'a') as f:
        for idx, i in enumerate(corr_list):
            tmp = str(idx) + ',' + str(i) + '\n'
            f.write(tmp)

            
