from sklearn import datasets
import train
import knn as kn
import numpy as np
import torch
import nns
import truth_improve
import utils

if __name__ == "__main__":
    dataset = 'Ham'
    data, test = utils.loadinDATASETS(dataset)
    np.random.shuffle(data)
    print(data.shape)
    print(test.shape)
    rowSize = data.shape[0]
    trainSize = int(rowSize / 1.6)
    train_x, train_y = data[:trainSize,0:-1], data[:trainSize,-1]
    test_x, test_y = test[:,0:-1], test[:,-1]
    res = np.zeros((102,))

    i = 2
    while i < 100:
        print('current iteration: ' + str(i))
        train_data = truth_improve.modefy_simple_sub_t(data[:trainSize, :-1],data[:trainSize,-1], times = i,zscore=False)
        #print(train_data.shape, test_x.shape[0])

        model = train.train(train_data[:,:-1], train_data[:,-1], nns.Ou_subNet)
        np.random.shuffle(test)
        test_x, test_y = test[:,0:-1], test[:,-1]
        #predict_sub
        tmp = 0
        nums = test_x.shape[0]
        print('predicting...')
        
        for j in range(nums):
            #print(j)
            if kn.predict_net_sub(test_x[j], train_x, train_y, 3, model) == test_y[j]:
                tmp += 1
            if j % 100 == 0:
                print('currently in '+ str(j))
        #print(res)
        for j in range(2):
            res[i+j] = tmp/nums
        print(tmp/nums)
        i += 2
        del model
    
        np.savetxt( dataset + 'valChangeRes_SBD.csv', res, delimiter=',')
