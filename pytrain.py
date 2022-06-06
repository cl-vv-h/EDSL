import train
import knn as kn
import numpy as np
import torch
import nns
import utils
import dis_com
import truth_improve

if __name__ == "__main__":
    datasets = ['Car', 'Earthquakes', 'Computers', 'CBF', 'DistalPhalanxOutlineCorrect', 'ChlorineConcentration', 'Ham', 'Haptics', 'MedicalImages', 'uWaveGestureLibrary_X']
    for dataset in datasets:
    #dataset = 'Computers'
        train_data = np.load('data/'+dataset+'.npy')
        #train_data = train.data_build(data[:, :-1])
        size = train_data.shape[1]

        model = train.train(train_data[:,:-1], train_data[:,-1], nns.Ou_subNet, size)

        torch.save(model, 'models/' + dataset + '_linear4_10_20_5_sub_ous20.pkl')


