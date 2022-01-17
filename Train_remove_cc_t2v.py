import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import math
import numpy as np
# import tensorflow as tf
import os
import time
import random
import dataprocessing as dp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from remove_cc_t2v import Model_3D
from parts_cp import dataset_times
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--startLead", type=int, default=1)
    parser.add_argument("--endLead", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--input", type=int, default=4)
    args = parser.parse_args()

    GPU_NUM = args.gpu
    device = torch.device('cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # Set a Hyper-parameters
    # regularizer_rate = 0.0    #L2 regularization
    batch_size = 64            # batch size
    ENS_Start = 0               # Starting No.
    ENS = 1                     # No. Ensemble Models
    numEpoch =  100             # No. Epoch
    learning_rate = 0.0001       # Initial Learning Rate
    n_cycles = 1                # No. cycles in Cosine Annealing
    epochs_per_cycle = math.floor(numEpoch / n_cycles)  # No. epochs for each cycle

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
    Folder = './local/' + os.path.abspath(__file__).split('\\')[-1].split('.')[0]
    pretrainFolder = ""
    dataFolder = 'c:/code/ENSO/ENSO_Ham/local/Dataset' #"./"

    SSTFile_train = dataFolder+'/Ham/cmip5_tr.input.1861_2001.nc'
    SSTFile_train_label = dataFolder+'/Ham/cmip5_tr.label.1861_2001.nc'
    SSTFile_train2 = dataFolder+'/Ham/cmip5_val.input.1861_2001.nc'
    SSTFile_train_label2 = dataFolder+'/Ham/cmip5_val.label.1861_2001.nc'
    
#     # Dataset for finetuning
#     Folder = "./finetuning_3DRFB_Attention"
#     pretrainFolder = "./pretrain_3DRFB_Attention_3"
#     dataFolder = "./"
    
    # SSTFile_val = dataFolder+'/Data_fine_tuning/soda.input.1871_1970.nc'
    # SSTFile_val_label = dataFolder+'/Data_fine_tuning/soda.label.1871_1970.nc'
    SSTFile_val = dataFolder+'/Ham/godas.input.1980_2017.nc'
    SSTFile_val_label = dataFolder+'/Ham/godas.label.1980_2017.nc'

    

    torch.cuda.empty_cache()
    for lead in range(args.startLead, args.endLead) :
        print('----------------------{}---------------------'.format(lead))
        # For updating learning rate
        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=iterations)
        l1 = nn.SmoothL1Loss()
        ent = nn.CrossEntropyLoss()

        for ens in range(ENS_Start, ENS) :
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            np.random.seed(ens)
            random.seed(ens)
            torch.manual_seed(ens)
            torch.cuda.manual_seed(ens)
            torch.cuda.manual_seed_all(ens)

            # Dataset for training
            trainset = dataset_times(SSTFile_train, SSTFile_train_label, lead, sstName='sst', hcName='t300', labelName='pr', scalar_time = True)  #datasets_general_3D_alllead_add(SSTFile_train, SSTFile_train_label, SSTFile_train2, SSTFile_train_label2, lead, sstName='sst', hcName='t300', labelName='pr', noise = True) 
            valset = dataset_times(SSTFile_val, SSTFile_val_label, lead, sstName='sst', hcName='t300', labelName='pr', scalar_time = True)

            eta_max = learning_rate     # Maximum laerning rate for Cosine Annealing
            eta_min = eta_max/100.0      # Minimum learning rate for Cosine Annealing
            
            trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=False)
            testloader = DataLoader(valset, batch_size = batch_size, shuffle=False)
            total_step = len(trainloader)
            test_step = len(testloader)


            print('{}/{}'.format(ens, ENS))

            model = Model_3D(2, noF, num_layer, num_answer, dr, args.input).to(device)
            # model = nn.DataParallel(model, device_ids=[0,1])
            # print(model)
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=reg, betas=(0.9, 0.999))
            # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=reg)

            if not os.path.exists("{}_{}/train_{}_{}/".format(Folder, args.input, lead, ens)):
                os.makedirs("{}_{}/train_{}_{}/".format(Folder, args.input, lead, ens), exist_ok=True)
            writer = SummaryWriter("{}_{}/Eval_{}_{}/".format(Folder, args.input, lead, ens))
            print('----------------lead : {}_{}---------------------------'.format(lead, ens))
            if pretrainFolder != '':
                model.load_state_dict(torch.load("{}/train_{}_{}/train_{}_{}.pth".format(pretrainFolder, lead, ens, lead, ens)))

            # Training
            loss_comp = 9999


            for epoch in range(numEpoch):

                trainloader = tqdm.tqdm(trainloader, desc = 'train', total=total_step)
                testloader = tqdm.tqdm(testloader, desc = 'val', total=test_step)

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
                for i, (batch, ansnino, anstype, t) in enumerate(trainloader):
                    # print(ansnino)
                    batch = Variable(batch.float().cuda())
                    ansnino = Variable(ansnino.float().cuda())
                    anstype = Variable(anstype.float().cuda())
                    t = t.unsqueeze(1).float().to(device='cuda')

                    # import matplotlib.pyplot as plt
                    # plt.imshow(batch[0, 0, :, :, 0].cpu())
                    # plt.colorbar()
                    # plt.savefig('map.png')
                    # plt.show()
                    # exit()
                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=True): 
                        output = model(batch, t)
                        prednino = np.squeeze(output[0], axis=2)
                        mseLoss = l1(prednino, ansnino)
                        pLoss = pearson(prednino, ansnino)
                        if (torch.isnan(pLoss)):
                            pLoss = 0
                        # predType = output[1] #torch.argmax(output[1], dim=1,keepdim=True)
                        anstype = torch.argmax(anstype, dim=1)
                        # entLoss = ent(predType, anstype)
                        
                        loss = mseLoss*0.0 + pLoss*1.0
                        losses[0] += mseLoss
                        losses[1] += pLoss
                        sum_train += loss
                        
                    scaler.scale(loss).backward() 
                    # scaler.scale(mseLoss*0.8+pLoss*0.2).backward(retain_graph=True)
                    # # scaler.scale().backward(retain_graph=True)
                    # scaler.scale(entLoss*0.2).backward()
                    scaler.step(optimizer) 
                    scaler.update()
                    # Backward and optimize
                    # loss.backward()
                    # optimizer.step()

                    del batch
                    del ansnino
                    del anstype
                    # print(losses)

                    trainloader.set_description(f'train loss : {loss:.4f}')

                trainloader.close()

                print('[{}/{} , {}/{} loss : {}, {}, {}'.format(epoch, numEpoch, (i+1), len(trainloader), losses[0]/len(trainloader), losses[1]/len(trainloader), losses[2]/len(trainloader)))
                writer.add_scalar('loss/train', sum_train/len(trainloader), epoch)
                # update_lr(optimizer, lr)

                model.eval()
                with torch.no_grad() :
                    for i, (batch, ansnino, anstype, t) in enumerate(testloader):
                        batch = Variable(batch.float().cuda())
                        ansnino = Variable(ansnino.float().cuda())
                        anstype = Variable(anstype.float().cuda())
                        t = t.unsqueeze(1).float().to(device='cuda')

                        output = model(batch, t)
                        prednino = np.squeeze(output[0], axis=2)
                        # prednino = np.squeeze(output, axis=1)
                        mseLoss = l1(prednino, ansnino)
                        pLoss = pearson(prednino, ansnino)
                        if (torch.isnan(pLoss)):
                            pLoss = 0

                        # predType = output[1] #torch.argmax(output[1], dim=1,keepdim=True)
                        # print(predType.shape)
                        anstype = torch.argmax(anstype, dim=1)
                        # print(anstype.shape)
                        # entLoss = ent(predType, anstype)
                        # print(output[0], ansnino)
                        loss = mseLoss*0.0 + pLoss*1.0
                        sum_test += loss
                        del batch
                        del ansnino
                        del anstype
                        del t

                        testloader.set_description(f'val loss : {loss:.4f}')


                    testloader.close()

                print('[{}/{}, val loss: {}'.format(epoch, numEpoch, sum_test/len(testloader)))
                if (sum_test/len(testloader)) < loss_comp : 
                    loss_comp = sum_test/len(testloader)
                    torch.save(model.state_dict(), "{}_{}/train_{}_{}/train_{}_{}.pth".format(Folder, args.input, lead, ens, lead, ens))
                    print('-'*50)
                    print('[{}/{} , loss_comp : {}'.format(epoch, numEpoch, loss_comp))
                    print('-'*50)
                    writer.add_scalar('loss/test', sum_test/len(testloader), epoch)
                    writer.flush()
            writer.close()
            # gc.collect()
