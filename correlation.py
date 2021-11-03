import dataprocessing as dp
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def datagraph(answer, pred, lead, filename, period) : 
    plt.clf()
    plt.plot(answer, label = 'lead time : {} answer graph_{}'.format(lead, period))
    plt.legend(loc = 'upper right')
    plt.plot(pred, label = 'lead time : {} prediction graph_{}'.format(lead, period))
    plt.legend(loc = 'upper right')
    plt.savefig('{}/datagraph{}_{}.png'.format(filename, lead, period))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correlation skill') 
    parser.add_argument("--ENS", type=int, default=10)
    parser.add_argument("--folder", type=str, default="Finetuning_3DResNet_GODAS")
    parser.add_argument("--startidx", type=int, default=15)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--name", type=str, default='')

    args = parser.parse_args()
    name = args.name 
    corrarray = np.zeros((args.count,))
    rmsearray = np.zeros((args.count,))
    uncarray = np.zeros((args.count,))
    uncarrayType = np.zeros((args.count,))
    accarray = np.zeros((args.count,))
    aucarray = np.zeros((args.count,))
    for idx, lead in enumerate(range(args.startidx, args.startidx + args.count)) :
        # predNino = np.load('{}/lead_{}_assemble_pred_nino_expanded.npy'.format(args.folder, lead))
        predNino = np.load('{}/lead_{}_assemble_pred_nino{}.npy'.format(args.folder, lead, name))
        assemble_pred_nino = np.mean(predNino, axis=0)
        # realNino = np.load('{}/lead_{}_assemble_real_nino_expanded.npy'.format(args.folder, lead))
        realNino = np.load('{}/lead_{}_assemble_real_nino{}.npy'.format(args.folder, lead, name))

        assemble_real_nino = np.mean(realNino, axis=0)
        corrarray[idx] = dp.CorrelationSkill(assemble_real_nino, assemble_pred_nino)

        # RMSE
        rmsearray[idx] = np.sqrt(np.mean((assemble_real_nino-assemble_pred_nino)**2))

        # Uncertainty
        # uncertainty = np.load('{}/lead_{}_assemble_uncertainty_nino_expanded.npy'.format(args.folder, lead))
        uncertainty = np.load('{}/lead_{}_assemble_uncertainty_nino{}.npy'.format(args.folder, lead, name))
        uncarray[idx] = np.mean(uncertainty)


        # type
        # realType = np.load('{}/lead_{}_assemble_real_type_expanded.npy'.format(args.folder, lead))
        realType = np.load('{}/lead_{}_assemble_real_type{}.npy'.format(args.folder, lead, name))

        assemble_real_type = np.mean(realType, axis=0)

        # predType = np.load('{}/lead_{}_assemble_pred_type_expanded.npy'.format(args.folder, lead))
        predType = np.load('{}/lead_{}_assemble_pred_type{}.npy'.format(args.folder, lead, name))

        assemble_pred_type = np.mean(predType, axis=0)
        predType_int = assemble_pred_type + 0.5
        predType_int = predType_int.astype(np.int32)

        accarray[idx] = accuracy_score(assemble_real_type, predType_int)
        # aucarray[idx] = roc_auc_score(assemble_real_type, assemble_pred_type)

        # uncertaintyType = np.load('{}/lead_{}_assemble_uncertainty_type_expanded.npy'.format(args.folder, lead))
        uncertaintyType = np.load('{}/lead_{}_assemble_uncertainty_type{}.npy'.format(args.folder, lead, name))
        uncarrayType[idx] = np.mean(uncertaintyType)
        
        datagraph(assemble_real_nino, assemble_pred_nino, lead, args.folder, args.name)
    np.savetxt('{}/correlation{}.csv'.format(args.folder, name), corrarray, delimiter=',')
    np.savetxt('{}/rmse{}.csv'.format(args.folder, name), rmsearray, delimiter=',')
    np.savetxt('{}/uncertainty{}.csv'.format(args.folder, name), uncarray, delimiter=',')
    np.savetxt('{}/type_uncertainty{}.csv'.format(args.folder, name), uncarrayType, delimiter=',')
    np.savetxt('{}/type_accuracy{}.csv'.format(args.folder, name), accarray, delimiter=',')
    # np.savetxt('{}/type_auc.csv'.format(args.folder), aucarray, delimiter=',')

    # np.savetxt('{}/correlation_expanded.csv'.format(args.folder), corrarray, delimiter=',')
    # np.savetxt('{}/rmse_expanded.csv'.format(args.folder), rmsearray, delimiter=',')
    # np.savetxt('{}/uncertainty_expanded.csv'.format(args.folder), uncarray, delimiter=',')
    # np.savetxt('{}/type_uncertainty_expanded.csv'.format(args.folder), uncarrayType, delimiter=',')
    # np.savetxt('{}/type_accuracy_expanded.csv'.format(args.folder), accarray, delimiter=',')