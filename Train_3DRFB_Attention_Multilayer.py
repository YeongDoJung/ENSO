import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import math
import imp
import numpy as np
import tensorflow as tf
import os
import time
import dataprocessing as dp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Models_3DRFB_Attention_Multilayer import Model_3D
from Parts import *
import argparse

def pearson(pred, gt):
    allLoss = 0
    for i in range(pred.shape[0]):
        score = pred[i, :]
        target = gt[i, :]
        vx = score - torch.mean(score)
        vy = target - torch.mean(target)
        loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        allLoss += 1.0 - loss
    allLoss /= pred.shape[0]
    return allLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--startLead", type=int, default=1)
    parser.add_argument("--endLead", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input", type=int, default=3)
    args = parser.parse_args()

    GPU_NUM = args.gpu
    device = torch.device('cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # Set a Hyper-parameters
    # regularizer_rate = 0.0    # L2 regularization
    batch_size = 512            # batch size
    ENS_Start = 0               # Starting No.
    ENS = 1                     # No. Ensemble Models
    numEpoch =  100             # No. Epoch
    learning_rate = 0.00001        # Initial Learning Rate
    n_cycles = 1                # No. cycles in Cosine Annealing
    epochs_per_cycle = math.floor(numEpoch / n_cycles)  # No. epochs for each cycle

    dr = 0.0                    # Dropout rate for Bayesian learning
    tau = 1.0                   # Weight for the batch size in regularization weight calculation (Bayesian learning)
    lengthscale = 0.01          # Default regularization weight (L2)
    reg = lengthscale**2 * (1 - dr) / (2. * batch_size * tau) # L2 regularization weight for Bayesian learning
    noF = 16                    # Initial No. filters

    num_layer =  128            # Feature size of 1st fully-connected layer
    num_answer = 2              # No. answers(3=3.4/ep/cp)

    minRMSE = 100.0             # minimum RMSE
    minUncertainty = 100.0      # minimum uncertainty

    leadMax = 24                # No. lead time

    # Dataset for pretraining
    Folder = "./pretrain_3DRFB_Attention_Multilayer"
    pretrainFolder = ""
    dataFolder = "./"

    SSTFile_train = dataFolder+'/Data_pre_training/cmip5_tr.input.1861_2001.nc'
    SSTFile_train_label = dataFolder+'/Data_pre_training/cmip5_tr.label.1861_2001.nc'
    # SSTFile_val = dataFolder+'/Data_pre_training/cmip5_val.input.1861_2001.nc'
    # SSTFile_val_label = dataFolder+'/Data_pre_training/cmip5_val.label.1861_2001.nc'
    
#     # Dataset for finetuning
#     Folder = "./finetuning_3DRFB_Attention"
#     pretrainFolder = "./pretrain_3DRFB_Attention_3"
#     dataFolder = "./"
    
    SSTFile_val = 'Data_fine_tuning/soda.input.1871_1970.nc'
    SSTFile_val_label = 'Data_fine_tuning/soda.label.1871_1970.nc'
#     SSTFile_val = dataFolder+'/Data_validation/godas.input.1980_2017.nc'
#     SSTFile_val_label = dataFolder+'/Data_validation/godas.label.1980_2017.nc'

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    for lead in range(args.startLead, args.endLead) :
        print('----------------------{}---------------------'.format(lead))
        # Dataset for training
        trainset = datasets_general_3D_alllead(SSTFile_train, SSTFile_train_label, lead, sstName='sst', hcName='t300', labelName='pr', noise = True) 
        valset = datasets_general_3D_alllead(SSTFile_val, SSTFile_val_label, lead, sstName='sst', hcName='t300', labelName='pr', noise = False)


        eta_max = learning_rate     # Maximum laerning rate for Cosine Annealing
        eta_min = eta_max/100.0      # Minimum learning rate for Cosine Annealing
        
        trainloader = DataLoader(trainset, batch_size= batch_size, shuffle=True)
        testloader = DataLoader(valset, batch_size= batch_size, shuffle=False)
        total_step = len(trainloader)
        test_step = len(testloader)

        # For updating learning rate
        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=iterations)
        # mse = nn.MSELoss()
        l1 = nn.SmoothL1Loss()
        # mse = nn.MSELoss()
        # cosine = nn.CosineSimilarity()
        ent = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        for ens in range(ENS_Start, ENS) :
            print('{}/{}'.format(ens, ENS))

            model = Model_3D(2, noF, num_layer, num_answer, dr, args.input).to(device)
            print(model)
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=reg, betas=(0.9, 0.999))

            if not os.path.exists("{}_{}/train_{}_{}/".format(Folder, args.input, lead, ens)):
                os.makedirs("{}_{}/train_{}_{}/".format(Folder, args.input, lead, ens))
            writer = SummaryWriter("{}_{}/Eval_{}_{}/".format(Folder, args.input, lead, ens))
            print('----------------lead : {}_{}---------------------------'.format(lead, ens))
            if pretrainFolder != '':
                model.load_state_dict(torch.load("{}/train_{}_{}/train_{}_{}.pth".format(pretrainFolder, lead, ens, lead, ens)))

            # Training
            loss_comp = 9999

            for epoch in range(numEpoch):
                sum_test = 0
                sum_train = 0
                
                realvalues = []
                predvalues = []
                uncertainty = []
                cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
                lr = eta_min + (eta_max - eta_min) * (1 + math.cos(cos_inner)) / 2.0

                # Training
                model.train()
                losses = np.zeros(3)
                for i, (batch, ansnino, anstype) in enumerate(trainloader):
                    # print(ansnino)
                    batch = Variable(batch.float().cuda())
                    ansnino = Variable(ansnino.float().cuda())
                    anstype = Variable(anstype.float().cuda())

                    # import matplotlib.pyplot as plt
                    # plt.imshow(batch[0, 0, :, :, 0].cpu())
                    # plt.colorbar()
                    # plt.savefig('map.png')
                    # plt.show()
                    # exit()
                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=True): 
                        output = model(batch)
                        prednino = np.squeeze(output[0], axis=2)
                        mseLoss = l1(prednino, ansnino)
                        pLoss = pearson(prednino, ansnino)
                        if (torch.isnan(pLoss)):
                            pLoss = 1.0
                        predType = output[1] #torch.argmax(output[1], dim=1,keepdim=True)
                        anstype = torch.argmax(anstype, dim=1)
                        entLoss = ent(predType, anstype)
                        
                        loss = mseLoss*0.8 + entLoss*0.2 + pLoss*0.2
                        losses[0] += mseLoss
                        losses[1] += pLoss
                        losses[2] += entLoss
                        sum_train += loss
                        
                    scaler.scale(loss).backward() 
                    scaler.step(optimizer) 
                    scaler.update()
                    # Backward and optimize
                    # loss.backward()
                    # optimizer.step()

                    del batch
                    del ansnino
                    del anstype
                print('[{}/{} , {}/{} loss : {}, {}, {}'.format(epoch, numEpoch, (i+1), len(trainloader), losses[0]/len(trainloader), losses[1]/len(trainloader), losses[2]/len(trainloader)))
                writer.add_scalar('loss/train', sum_train/len(trainloader), epoch)
                # update_lr(optimizer, lr)

                model.eval()
                with torch.no_grad() :
                    for i, (batch, ansnino, anstype) in enumerate(testloader):
                        batch = Variable(batch.float().cuda())
                        ansnino = Variable(ansnino.float().cuda())
                        anstype = Variable(anstype.float().cuda())

                        output = model(batch)
                        prednino = np.squeeze(output[0], axis=2)
                        # prednino = np.squeeze(output, axis=1)
                        mseLoss = l1(prednino, ansnino)
                        pLoss = pearson(prednino, ansnino)
                        if (torch.isnan(pLoss)):
                            pLoss = 1.0
                        predType = output[1] #torch.argmax(output[1], dim=1,keepdim=True)
                        # print(predType.shape)
                        anstype = torch.argmax(anstype, dim=1)
                        # print(anstype.shape)
                        entLoss = ent(predType, anstype)
                        # print(output[0], ansnino)
                        loss = mseLoss*0.8 + entLoss*0.2 + pLoss*0.2
                        # loss = entLoss
                        sum_test += loss
                        del batch
                        del ansnino
                        del anstype

                print('[{}/{}, val loss: {}'.format(epoch, numEpoch, sum_test/len(testloader)))
                if (sum_test/len(testloader)) < loss_comp : 
                    loss_comp = sum_test/len(testloader)
                    torch.save(model.state_dict(), "{}_{}/train_{}_{}/train_{}_{}.pth".format(Folder, args.input, lead, ens, lead, ens))
                    print('[{}/{} , loss_comp : {}'.format(epoch, numEpoch, loss_comp))
                    writer.add_scalar('loss/test', sum_test/len(testloader), epoch)
                    writer.flush()
            writer.close()
            gc.collect()
