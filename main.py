import train
import knn
from keras.datasets import boston_housing
import numpy as np
import torch

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    #x_train, y_train = x_train[0:100], y_train[0:100]
    x_train = train.data_build(x_train)
    np.random.shuffle(x_train)
    y_train = x_train[:,-1] 
    x_train = x_train[:,0:-1]

    print(y_train)
    model = torch.load('model_ou.pkl')

    
