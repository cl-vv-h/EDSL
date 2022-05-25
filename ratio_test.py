from sklearn import datasets
import train
import knn as kn
import numpy as np
import torch
import nns
import truth_improve
import utils
import dis_com
datasets = ['Car', 'Earthquakes', 'Computers', 'CBF', 'DistalPhalanxOutlineCorrect', 'ChlorineConcentration', 'Ham', 'Haptics', 'MedicalImages', 'uWaveGestureLibrary_X']
res = []
for dataset in datasets:
    print(dataset + ' is generating...')
    data, test = utils.loadinDATASETS(dataset)
    np.random.shuffle(data)
    #print(data.shape)
    trainSize = min(data.shape[0], 80)

    train_x, train_y = data[:trainSize,0:-1], data[:trainSize,-1]
    test_x, test_y = test[:,0:-1], test[:,-1]

    train_data = truth_improve.modefy_simple_sub_ratio(data[:trainSize, :-1],data[:trainSize,-1], dis_com.ou_dis, zscore=False)
    print(train_data.shape)

    hidden_size = min(train_data.shape[1], 500)
    model = train.train(train_data[:,:-1], train_data[:,-1], nns.Ou_subNet, hidden_size)
    np.random.shuffle(test)
    test_x, test_y = test[:,0:-1], test[:,-1]
    #predict_sub
    tmp = 0
    nums = min(test_x.shape[0], 300)
    print('predicting...')
    
    for j in range(nums):
        #print(j)
        if kn.predict_net_sub(test_x[j], train_x, train_y, 3, model) == test_y[j]:
            tmp += 1
        if j % 100 == 0:
            print('currently in '+ str(j))
    #print(res)
    res.append(tmp/nums)
    print('accuracy is '+ str(tmp/nums))

    del model
np.array(res, dtype=float)
np.savetxt('ratioResou.csv', res, delimiter=',')
