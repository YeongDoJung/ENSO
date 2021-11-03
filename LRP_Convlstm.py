import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np 
from Models_Convlstm import Model_3D
from Parts import *
import utils
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Folder = "Train_3D_ResBlock_prof3_73_"
num_layer = 128
num_answer = 1
noF = 8  
dr = 0.01
model = Model_3D(2, noF, num_layer, num_answer, dr).to(device)
leadtime = 24 
ENS = 10
for lead in [6, 12, 18, 24] :
    testset = testdataset_godas_not2d(lead, 1, 3)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    for m, (img_prop, _) in enumerate(testloader) :
        ensembleR = [[] for x in range(ENS)]    
        for ens in range(ENS) : 
            model.load_state_dict(torch.load('{}/train_{}_{}/train_{}_{}.pth'.format(Folder, lead, ens , lead, ens)))
            model.eval()
            layers = []  
            trainStartIdx = 0 
            for i in model._modules.keys() :
                layers.append(model._modules[i])
            layers = layers[2:]
            L = len(layers)
            print('[lead time : [{}/{}], test time :[{}/{}]'.format(lead, leadtime, (m+1), len(testloader)))
            img_prop = Variable(img_prop.float().cuda())
            A = [img_prop]+[None]*L
            for l in range(L): 
                A[l+1] = layers[l].forward(A[l])
            R = [None]*L + [(A[-1]).data]
            for l in range(0, L)[::-1]:
                A[l] = (A[l].data).requires_grad_(True)
                rho = lambda p: p
                incr = lambda z: z + 1e-9
                z = layers[l].forward(A[l])  # step 1
                s = (R[l+1]/z).data                                    # step 2
                (z*s).sum().backward()
                c = A[l].grad                  # step 3
                R[l] = (A[l]*c).data                                   # step 4
            if ens == 0 :
                base = R[0].cpu()
            else :
                base = np.concatenate((base, R[0].cpu()))
        LRPmap = np.mean(base, axis = 0) 
        del base
        for k in range(2) :
            for j in range(3) :
                x = np.float32(LRPmap[k, j, :, :])
                # SST Map
                if k == 0 :
                    if not os.path.exists('lrpassemble/{}/train_{}/sst_{}y_{}m/'.format(Folder, lead, (2011)+(m//12), (m%12)+1)):
                        os.makedirs('lrpassemble/{}/train_{}/sst_{}y_{}m/'.format(Folder, lead, (2011)+(m//12), (m%12)+1))
                    utils.heatmap(x, Folder, lead, m ,j, sx = 10, sy = 5)
                    np.save('lrpassemble/{}/train_{}/sst_{}y_{}m/{}'.format(Folder, lead, (2011)+(m//12), (m%12)+1, (j+1)),x)
                # HC Map / prof3 setting에선 0으로 입력값이 들어가서 전부 0이 나옴
                # if k == 1 :
                #     if not os.path.exists('lrpassemble/{}/train_{}/hc_{}y_{}m/'.format(Folder, lead, (2011)+(m//12), (m%12)+1)):
                #         os.makedirs('lrpassemble/{}/train_{}/hc_{}y_{}m/'.format(Folder, lead, (2011)+(m//12), (m%12)+1))
                #     utils.heatmap(x, Folder, lead, m ,j, sx = 10, sy = 5)
                #     np.save('lrpassemble/{}/train_{}/hc_{}y_{}m/{}'.format(Folder, lead, (2011)+(m//12), (m%12)+1, (j+1)),x)
            
