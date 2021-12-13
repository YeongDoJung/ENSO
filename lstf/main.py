import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import math
import numpy as np
import os
import sys
sys.path.append('.')
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import model
from datasets.basicdatasets import basicdataset
import argparse
import tqdm

def train(args, model, trainloader, testloader):
    args = args
    sum_test = 0
    sum_train = 0

    # cosine annealing
    # eta_max = learning_rate     # Maximum laerning rate for Cosine Annealing
    # eta_min = eta_max/100.0      # Minimum learning rate for Cosine Annealing
    # cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
    # lr = eta_min + (eta_max - eta_min) * (1 + math.cos(cos_inner)) / 2.0

    # Training
    model.train()
    losses = np.zeros(3)
    for i, (batch, ansnino) in enumerate(trainloader):
        # print(ansnino)
        batch = Variable(batch.float().cuda())
        tgt = torch.zeros_like(batch).cuda()
        ansnino = Variable(ansnino.float().cuda())

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True): 
            output = model(batch)

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
            
        scaler.scale(loss).backward() 
        scaler.step(optimizer) 
        scaler.update()

        del batch
        del ansnino

    print(f'[{args.current_epoch}/{numEpoch} , {i+1}/{len(trainloader)} loss : {losses[0]/len(trainloader)}, {losses[1]/len(trainloader)}, {losses[2]/len(trainloader)}')
    writer.add_scalar('loss/train', sum_train/len(trainloader), epoch)

    model.eval()
    with torch.no_grad() :
        for i, (batch, ansnino, anstype) in enumerate(testloader):
            batch = Variable(batch.float().cuda())
            ansnino = Variable(ansnino.float().cuda())

            output = model(batch)
            # prednino = np.squeeze(output[0], axis=2)
            # # prednino = np.squeeze(output, axis=1)
            # mseLoss = l1(prednino, ansnino)
            # pLoss = pearson(prednino, ansnino)
            # if (torch.isnan(pLoss)):
            #     pLoss = 0

            # predType = output[1] #torch.argmax(output[1], dim=1,keepdim=True)
            # # print(predType.shape)
            # anstype = torch.argmax(anstype, dim=1)
            # # print(anstype.shape)
            # entLoss = ent(predType, anstype)
            # # print(output[0], ansnino)
            # loss = mseLoss*0.0 + entLoss*0.2 + pLoss*0.8
            # sum_test += loss
            # del batch
            # del ansnino

    print('[{}/{}, val loss: {}'.format(epoch, numEpoch, sum_test/len(testloader)))
    if (sum_test/len(testloader)) < loss_comp : 
        loss_comp = sum_test/len(testloader)
        torch.save(model.state_dict(), "{}_{}/train_{}_{}/train_{}_{}.pth".format(Folder, args.input, lead, ens, lead, ens))
        print('[{}/{} , loss_comp : {}'.format(epoch, numEpoch, loss_comp))
        writer.add_scalar('loss/test', sum_test/len(testloader), epoch)
        writer.flush()
    writer.close()
    # gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--startLead", type=int, default=1)
    parser.add_argument("--endLead", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input", type=int, default=3)
    args = parser.parse_args()

    print(torch.cuda.is_available())

    GPU_NUM = args.gpu
    device = torch.device('cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # Set a Hyper-parameters
    # regularizer_rate = 0.0    #L2 regularization
    batch_size = 16            # batch size
    ENS_Start = 0               # Starting No.
    ENS = 10                     # No. Ensemble Models
    numEpoch =  10             # No. Epoch
    learning_rate = 0.0001       # Initial Learning Rate

    dr = 0.0                    # Dropout rate for Bayesian learning
    tau = 1.0                   # Weight for the batch size in regularization weight calculation (Bayesian learning)
    lengthscale = 0.1           # Default regularization weight (L2)
    reg = lengthscale**2 * (1 - dr) / (2. * batch_size * tau) # L2 regularization weight for Bayesian learning
    noF = 16                     # Initial No. filters

    num_layer =  256             # Feature size of 1st fully-connected layer
    num_answer = 2              # No. answers(3=3.4/ep/cp)

    minRMSE = 100.0             # minimum RMSE
    minUncertainty = 100.0      # minimum uncertainty

    leadMax = 24                # No. lead time

    # Dataset for pretraining
    Folder = "./3DRFB_Transformer"
    dataFolder = './Dataset' #"./"

    SSTFile_train = dataFolder+'/Ham/cmip5_tr.input.1861_2001.nc'
    SSTFile_train_label = dataFolder+'/Ham/cmip5_tr.label.1861_2001.nc'
    SSTFile_val = dataFolder+'/Ham/godas.input.1980_2017.nc'
    SSTFile_val_label = dataFolder+'/Ham/godas.label.1980_2017.nc'

    # Dataset for training
    trainset = basicdataset(SSTFile_train, SSTFile_train_label, sstName='sst', hcName='t300', labelName='pr')  #datasets_general_3D_alllead_add(SSTFile_train, SSTFile_train_label, SSTFile_train2, SSTFile_train_label2, lead, sstName='sst', hcName='t300', labelName='pr', noise = True) 
    valset = basicdataset(SSTFile_val, SSTFile_val_label, sstName='sst', hcName='t300', labelName='pr')

    print(trainset._batchsize())
    exit()

    model = model.mdl.RFB_Transformer(in_channels, out_channels, d_model)

    torch.cuda.empty_cache()

    args.current_epoch = 0

    for epoch in range(numEpoch):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        criterion = nn.BCELoss()

        train(args, model, trainloader, testloader)
        test(args, model, testloader)








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
    #         scaler = torch.cuda.amp.GradScaler(enabled=True)
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

    #         if not os.path.exists("{}_{}/train_{}_{}/".format(Folder, args.input, lead, ens)):
    #             os.makedirs("{}_{}/train_{}_{}/".format(Folder, args.input, lead, ens))
    #         writer = SummaryWriter("{}_{}/Eval_{}_{}/".format(Folder, args.input, lead, ens))
    #         print('----------------lead : {}_{}---------------------------'.format(lead, ens))
    #         if pretrainFolder != '':
    #             model.load_state_dict(torch.load("{}/train_{}_{}/train_{}_{}.pth".format(pretrainFolder, lead, ens, lead, ens)))

    #         # Training
    #         loss_comp = 9999

            
