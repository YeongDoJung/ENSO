import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
from Models_3DResNet import Model_3D
from Parts import *
import utils
import os
import matplotlib.pyplot as plt
from theconf import Config as C, ConfigArgumentParser

random_seed = 722
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
np.random.seed(random_seed)
random.seed(random_seed)

if __name__ == "__main__":
    parser = ConfigArgumentParser(conflict_handler="resolve")
    parser.add_argument("--ENS", type=int, default=10)
    parser.add_argument("--folder", type=str, default="Train_3D_Pretraining_3DResNet")
    parser.add_argument("--type", type=str, default="_normal")
    parser.add_argument("--task", type=str, default="nino")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    pred = np.mean(np.load('{}/lead_9_assemble_pred_nino_2011~2015.npy'.format(args.folder)), axis = 0)
    real = np.mean(np.load('{}/lead_9_assemble_real_nino_2011~2015.npy'.format(args.folder)), axis = 0)
    plt.plot(pred, label = 'prediction')
    plt.plot(real, label = 'real')
    plt.legend(['prediction','real'])
    plt.savefig('test_3_9.png')
    exit()
    print('lead_9_assemble_real_nino_2015~2018.npy')

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    for lead in C.get()['leadtime'] :
        DataDir = 'Data_validation/'
        SSTFile = DataDir +'oisst_ssta_monthly.nc'
        HCFile = DataDir +'ecmwf_hca_monthly(1982-2015).nc'

        testset = test_datasets_general_3D(SSTFile, HCFile, lead, 1, 3, C.get()['startidx'], C.get()['endidx'], C.get()['ansdur'])
        testloader = DataLoader(testset, batch_size=1, shuffle=False)
        model = Model_3D(2, C.get()['noF'], C.get()['numlayer'], C.get()['numanswer'], C.get()['drop']).to(device)
        model.load_state_dict(torch.load('{}/train_{}_{}/train_{}_{}.pth'.format(args.folder, lead, 10 , lead, 10)))
        model.eval()
        for m, (img_prop, _, _) in enumerate(testloader) :
            print('[{}/{}]'.format(m, len(testloader)))
            batchList = []
            for j in range(100):
                batchList.append(img_prop)
            batch = torch.cat(batchList, dim=0)
            _LRP = utils.LRP_make(model, batch,'{}'.format(args.task))
            # 이상한 값들이 많아서 값을 날려줬음
            LRP = np.array(_LRP)
            LRP = LRP.astype(dtype=np.float32)
            LRP[LRP < -100] = np.nan
            LRP[LRP > 100] = np.nan
            LRP = np.nan_to_num(LRP, 0.0)
            LRP = np.mean(LRP, axis = 0) 
            for k in range(2) :
                for j in range(3) :
                    LRP_nino_ = LRP[k, j, :, :]
                    # SST Map
                    if k == 0 :
                        if not os.path.exists('{}_LRP{}_{}_modify/train_{}/sst_{}y_{}m/'.format(args.folder, args.type, args.task, lead, (2011)+(m//12), (m%12)+1)):
                            os.makedirs('{}_LRP{}_{}_modify/train_{}/sst_{}y_{}m/'.format(args.folder, args.type, args.task, lead, (2011)+(m//12), (m%12)+1))
                        utils.heatmap(LRP_nino_, args.folder, lead, m ,j, args.type, args.task, sx = 10, sy = 5)
                        np.save('{}_LRP{}_{}_modify/train_{}/sst_{}y_{}m/sst_{}y_{}m_{}'.format(args.folder, args.type, args.task,  lead, (2011)+(m//12), (m%12)+1, (2011)+(m//12), (m%12)+1, j), LRP_nino_)