import dataprocessing as dp
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from scipy.special import softmax

def datagraph(answer, pred, lead, filename) : 
    plt.clf()
    plt.plot(answer, label = 'lead time : {} answer graph'.format(lead))
    plt.legend(loc = 'upper right')
    plt.plot(pred, label = 'lead time : {} prediction graph'.format(lead))
    plt.legend(loc = 'upper right')
    plt.savefig('{}/datagraph{}.png'.format(filename, lead))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--ENS", type=int, default=10)
    parser.add_argument("--folder", type=str, default="Train_3D_Scratch_3DResNet")
    parser.add_argument("--folder2", type=str, default="Train_3D_Scratch_3DResNet")
    parser.add_argument("--startidx", type=int, default=15)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    corrarray = np.zeros((args.count,))
    rmsearray = np.zeros((args.count,))
    uncarray = np.zeros((args.count,))
    uncarrayType = np.zeros((args.count,))
    accarray = np.zeros((args.count,))
    aucarray = np.zeros((args.count,))

    for idx, lead in enumerate(range(args.startidx, args.startidx + args.count)) :
        print('[{}/{}]'.format(idx+1, args.count))
        predNino1 = np.load('{}/lead_{}_assemble_pred_nino.npy'.format(args.folder, lead))
        predNino2 = np.load('{}/lead_{}_assemble_pred_nino.npy'.format(args.folder2, lead))
        predNino = np.concatenate((predNino1, predNino2), axis = 0)

        realNino1 = np.load('{}/lead_{}_assemble_real_nino.npy'.format(args.folder, lead))
        realNino2 = np.load('{}/lead_{}_assemble_real_nino.npy'.format(args.folder2, lead))
        realNino = np.concatenate((realNino1, realNino2), axis = 0)

        # assemble_real_nino = np.mean(realNino, axis=0)
        # corrarray[idx] = dp.CorrelationSkill(assemble_real_nino, assemble_pred_nino)
        # # RMSE
        # rmsearray[idx] = np.sqrt(np.mean((assemble_real_nino-assemble_pred_nino)**2))

        # Uncertainty
        uncertainty1 = np.load('{}/lead_{}_assemble_uncertainty_nino.npy'.format(args.folder, lead))
        uncertainty2 = np.load('{}/lead_{}_assemble_uncertainty_nino.npy'.format(args.folder2, lead))
        uncertainty = np.concatenate((uncertainty1, uncertainty2), axis = 0)

        assemble_pred_nino = np.mean((predNino * (1 - softmax(uncertainty, axis = 0))), axis=0)
        assemble_real_nino = np.mean((realNino * (1 - softmax(uncertainty, axis = 0))), axis=0)
        corrarray[idx] = dp.CorrelationSkill(assemble_real_nino, assemble_pred_nino)


        # print(np.mean(uncertainty, axis=0), realNino[:, 0], assemble_pred_nino.shape)
        # exit()
        # uncarray[idx] = np.mean(uncertainty)

        # # type
        # assemble_real_type = np.mean(realType, axis=0)

        # type
        realType = np.load('{}/lead_{}_assemble_real_type.npy'.format(args.folder, lead))
        assemble_real_type = np.mean(realType, axis=0)

        predType = np.load('{}/lead_{}_assemble_pred_type.npy'.format(args.folder, lead))
        assemble_pred_type = np.mean(predType, axis=0)
        predType_int = assemble_pred_type + 0.5
        predType_int = predType_int.astype(np.int32)

        # accarray[idx] = accuracy_score(assemble_real_type, predType_int)
        # aucarray[idx] = roc_auc_score(assemble_real_type, assemble_pred_type)

        uncertaintyType = np.load('{}/lead_{}_assemble_uncertainty_type.npy'.format(args.folder, lead))
        uncarrayType[idx] = np.mean(uncertaintyType)

        datagraph(assemble_real_nino, assemble_pred_nino, lead, args.folder)

    np.savetxt('{}/correlation_softmax.csv'.format(args.folder), corrarray, delimiter=',')
    # np.savetxt('{}/rmse.csv'.format(args.folder), rmsearray, delimiter=',')
    # np.savetxt('{}/uncertainty.csv'.format(args.folder), uncarray, delimiter=',')
    # np.savetxt('{}/type_uncertainty.csv'.format(args.folder), uncarrayType, delimiter=',')
    # np.savetxt('{}/type_accuracy.csv'.format(args.folder), accarray, delimiter=',')
    # np.savetxt('{}/type_auc.csv'.format(args.folder), aucarray, delimiter=',')