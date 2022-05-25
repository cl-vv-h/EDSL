from imghdr import tests
import train
import knn
import numpy as np
import torch
import nns
import utils
import dis_com

if __name__ == "__main__":
    #data = utils.loadData('ECG5000')
    [train_data, test_data] = utils.loadinDATASETS('CBF')
    print(test_data.shape)

    testSize = 100
    np.random.shuffle(test_data)
    train_x, train_y = train_data[:,0:-1], train_data[:,-1]
    test_x, test_y = test_data[:,0:-1], test_data[:,-1]

    nums = test_x.shape[0]
    real = 0

    for i in range(nums):
        print(i)
        predict = knn.predict(test_x[i], train_x, train_y, 3, dis_com.cos)
        if predict == test_y[i]:
            real += 1
    
    print(real/nums)

    
