from imghdr import tests
from statistics import mode
import train
import knn as kn
import numpy as np
import torch
import nns
import utils
from sklearn.neighbors import KNeighborsClassifier
import dis_com

if __name__ == "__main__":
    #data = utils.loadData('Car')
    dataset = 'Computers'
    data,test = utils.loadinDATASETS(dataset)
    print(data.shape)
    np.random.shuffle(test)
    rowSize = data.shape[0]
    trainSize = rowSize // 2
    testSize =  test.shape[0]

    print(dataset + ' testSize: ', testSize)

    train_x, train_y = data[:,0:-1], data[:,-1]
    test_x, test_y = test[:,0:-1], test[:,-1]
    #predict_sub
    tmp = 0
    nums = test_x.shape[0]
    model = torch.load('models/' + dataset +'_linear4_10_20_5_sub_ous20.pkl')
    print(model)
    
    for i in range(nums):
        print(i)
        if kn.predict_net_sub(test_x[i], train_x, train_y, 3, model) == test_y[i]:
            tmp += 1
    #print(res)
    print(tmp/nums)




#2  0.825  0.7825  0.8625  0.8125  0.875   0.85
#3  0.8375  0.825   0.825   0.85    0.825
