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

from lstf import metric
from lstf.model import build
from lstf.datasets.basicdatasets import tdimdataset
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


def train(args, model, optimizer, trainset, valset, criterion):
    args = args
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if not os.path.exists(f'{Folder}/'):
        os.makedirs(f'{Folder}/')
    writer = SummaryWriter(f'{Folder}/eval_{args.current_epoch}')
    
    trainloader = tqdm.tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True), total=len(trainset)//args.batch_size)

    sum_test = 0
    sum_train = 0

    # cosine annealing
    # eta_max = learning_rate     # Maximum laerning rate for Cosine Annealing
    # eta_min = eta_max/100.0      # Minimum learning rate for Cosine Annealing
    # cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
    # lr = eta_min + (eta_max - eta_min) * (1 + math.cos(cos_inner)) / 2.0

    # Training
    model.train()

    trainloss = metric.AverageMeter()
    valloss = metric.AverageMeter()
    
    for i, (batch, ansnino) in enumerate(trainloader):
        # print(ansnino)
        batch = batch.clone().detach().requires_grad_(True).to(device=device)
        ansnino = ansnino.clone().detach().requires_grad_(True).to(device=device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True): 
            output = model(batch)
            tl = criterion(output, ansnino)
            trainloss.update(tl)

            # prednino = np.squeeze(output[0], axis=2)
            # mseLoss = l1(prednino, ansnino)
            # pLoss = pearson(prednino, ansnino)
            # if (torch.isnan(pLoss)):
            #     pLoss = 0
            # predType = output[1] #torch.argmax(output[1], dim=1,keepdim=True)
            
            # loss = mseLoss*0.0 + entLoss*0.2 + pLoss*0.8
            # losses[0] += mseLoss
            # losses[1] += pLoss
            # losses[2] += entLoss
            # sum_train += loss
            
        scaler.scale(tl).backward() 
        scaler.step(optimizer) 
        scaler.update()

        del batch
        del ansnino

        trainloader.set_description(f'{args.current_epoch}/{args.numEpoch} loss = {trainloss.avg:.4f}-{tl:.4f}')

    writer.add_scalar('loss/train', trainloss.avg, args.current_epoch)

    testloader = tqdm.tqdm(DataLoader(valset, batch_size=args.batch_size, shuffle=False), total=len(valset)//args.batch_size)
    model.eval()

    assemble_real_nino = np.zeros((len(valset), 23))
    assemble_pred_nino = np.zeros((len(valset), 23))

    with torch.no_grad() :
        for i, (batch, ansnino) in enumerate(testloader):
            batch = batch.clone().detach().requires_grad_(True).to(device=device)
            ansnino = ansnino.clone().detach().requires_grad_(True).to(device=device)

            idx = batch.shape[0]*i
            uncertaintyarry_nino = np.zeros((1, batch.shape[0], 23))

            for b in range(int(1)):
                output = model(batch) # inference
                vl = criterion(output, ansnino)
                valloss.update(vl)
                uncertaintyarry_nino[b, :, :] = output.cpu()

                assemble_real_nino[idx:idx+batch.shape[0], :] = ansnino.cpu().numpy()

            assemble_pred_nino[idx:idx+batch.shape[0], :] += np.mean(uncertaintyarry_nino, axis=0)
            
            del batch
            del ansnino

            testloader.set_description(f'{args.current_epoch}/{args.numEpoch} loss = {valloss.avg:.4f}-{vl:.4f}')

        
        corr = np.zeros(23)
        for i in range(23):
            corr[i] = metric.CorrelationSkill(assemble_real_nino[:, i], assemble_pred_nino[:, i])

        mse = mean_squared_error(assemble_pred_nino, assemble_real_nino)
        print(corr)

    corr_list.append(np.mean(corr))

    if (np.mean(corr)) > args.corr : 
        args.corr = np.mean(corr)
        os.makedirs(f'{Folder}/eval_{args.current_epoch}/', exist_ok=True)
        torch.save(model.state_dict(), f'{Folder}/eval_{args.current_epoch}/eval_{args.current_epoch}.pth')
        np.savetxt(f'{Folder}/eval_{args.current_epoch}/eval_{args.current_epoch}_acc_{mse:.4f}_corr.csv', corr)
        print('[{}/{} , mean corr : {}'.format(args.current_epoch, args.numEpoch, args.corr))
        writer.add_scalar('corr/test', np.mean(corr), args.current_epoch)
        writer.flush()
    writer.close()

    args.current_epoch += 1
    trainloss.reset()
    valloss.reset()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--startLead", type=int, default=1)
    parser.add_argument("--endLead", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--numEpoch", type=int, default=700)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--name", type=str, default='CNN_2D')


    parser.add_argument("--val_min", type=float, default=9999)
    parser.add_argument("--current_epoch", type=int, default=0)
    args = parser.parse_args()

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

    args.corr = 0

    # Dataset for pretraining
    Folder = Path(str(Path(__file__).parent) + "/local/" + args.name)
    dataFolder = Path(str(Path(__file__).parent) + '/local/Dataset/Ham/') #"./"


    SSTFile_train = dataFolder / 'cmip5_tr.input.1861_2001.nc'
    SSTFile_train_label = dataFolder / 'cmip5_tr.label.1861_2001.nc'
    SSTFile_val = dataFolder / 'godas.input.1980_2017.nc'
    SSTFile_val_label = dataFolder / 'godas.label.1980_2017.nc'

    # Dataset for training
    trainset = tdimdataset(SSTFile_train, SSTFile_train_label, sstName='sst', hcName='t300', labelName='pr')  #datasets_general_3D_alllead_add(SSTFile_train, SSTFile_train_label, SSTFile_train2, SSTFile_train_label2, lead, sstName='sst', hcName='t300', labelName='pr', noise = True) 
    valset = tdimdataset(SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')


    
    model = build.Model_2D().to(device=device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005, alpha=0.9)
    criterion = nn.SmoothL1Loss(reduction='mean')

    corr_list = []
    
    torch.cuda.empty_cache()

    for epoch in range(args.numEpoch):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train(args, model=model, optimizer=optimizer, trainset=trainset, valset=valset, criterion=criterion)
        # test(args, model=model, testloader)

    with open(Folder.joinpath('corr.csv'), 'a') as f:
        for idx, i in enumerate(corr_list):
            tmp = str(idx) + ',' + str(i) + '\n'




    # for lead in range(args.startLead, args.endLead) :
    #     print('----------------------{}---------------------'.format(lead))
    #     # For updating learning rate
    #     def update_lr(optimizer, lr):
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr

    #     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=iterations)
    #     l1 = nn.SmoothL1Loss()
    #     ent = nn.CrossEntropyLoss()

    #     for ens in range(ENS_Start, ENS) :
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    #         np.random.seed(ens)
    #         random.seed(ens)
    #         torch.manual_seed(ens)
    #         torch.cuda.manual_seed(ens)
    #         torch.cuda.manual_seed_all(ens)

    #         # Dataset for training
    #         trainset = basicdataset(SSTFile_train, SSTFile_train_label, sstName='sst', hcName='t300', labelName='pr')  #datasets_general_3D_alllead_add(SSTFile_train, SSTFile_train_label, SSTFile_train2, SSTFile_train_label2, lead, sstName='sst', hcName='t300', labelName='pr', noise = True) 
    #         valset = basicdataset(SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')


    #         trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=False)
    #         testloader = DataLoader(valset, batch_size = batch_size, shuffle=False)
    #         total_step = len(trainloader)
    #         test_step = len(testloader)

    #         print('{}/{}'.format(ens, ENS))

    #         model = Model_3D(2, noF, num_layer, num_answer, dr, args.input).to(device)
    #         print(model)
    #         optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=reg, betas=(0.9, 0.999))
    #         # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=reg)


    #         if pretrainFolder != '':
    #             model.load_state_dict(torch.load("{}/train_{}_{}/train_{}_{}.pth".format(pretrainFolder, lead, ens, lead, ens)))

    #         # Training
    #         loss_comp = 9999

            