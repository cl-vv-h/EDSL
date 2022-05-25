import train
import knn as kn
import numpy as np
import torch
import nns
import utils
import dis_com
import truth_improve

if __name__ == "__main__":
    dataset = 'Computers'
    train_data = np.loadtxt('trainData.csv', delimiter=',')
    #train_data = train.data_build(data[:, :-1])
    print(train_data.shape)

    model = train.train(train_data[:,:-1], train_data[:,-1], nns.Ou_subNet)

    torch.save(model, 'models/' + dataset + '_linear4_10_20_5_sub_ous20.pkl')


