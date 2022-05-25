from statistics import mode
import train
import knn as kn
import numpy as np
import torch
import nns
import utils
from sklearn.neighbors import KNeighborsClassifier
import dis_com
import truth_improve as ti

if __name__ == "__main__":
    data = utils.loadData('Car')
    np.random.shuffle(data)
    rowSize = data.shape[0]
    columnSize = data.shape[1] - 1
    trainSize = rowSize // 3
    testSize = rowSize - trainSize
    train_x, train_y = data[:trainSize,0:-1], data[:trainSize,-1]
    test_x, test_y = data[trainSize:,0:-1], data[trainSize:,-1]


    real_dis = train.data_build(test_x)
    truth = real_dis[:,-1]


    predict = np.zeros(truth.shape)

    model = torch.load('model_lstm_2_6_ou_sub_improved.pkl')

    real_dis_sub = real_dis[:,0:columnSize] - real_dis[:,columnSize:2*columnSize]
    test_cuda = utils.np2tensor(real_dis[:,:-1])
    test_cuda = test_cuda.cuda()

    test_sub_cuda = utils.np2tensor(real_dis_sub).cuda()

    for i in range(truth.shape[0]):
        print(i)
        predict[i] = model(test_sub_cuda[i]).cpu().detach().numpy()
    
    tmp = truth - predict


    np.savetxt('res.csv' , tmp, delimiter=',')
    np.savetxt('truth.csv' , truth, delimiter=',')
    np.savetxt('predict.csv' , predict, delimiter=',')





