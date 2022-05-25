import torch
import numpy as np
import pandas as pd

dataset = ['Car', 'Earthquakes', 'ECG5000', 'UWaveGestureLibraryX']

def np2tensor(x):
    x_ = torch.from_numpy(x)
    x_ = x_.to(torch.float32)
    return x_

def loadcsv2np(set_name):
    res = np.loadtxt('.\datasets\\' + set_name + '.csv', delimiter=',')
    return res

def loadboth2np(set_name):
    train = np.loadtxt('.\datasets\\' + set_name + '_TRAIN.csv', delimiter=',')
    test = np.loadtxt('.\datasets\\' + set_name + '_TEST.csv', delimiter=',')

    return np.concatenate((train, test))

def loadData(set_name):
    if set_name in dataset:
        return loadboth2np(set_name)
    else:
        return loadcsv2np(set_name)

def loadinDATASETS(set_name):
    train = np.loadtxt('.\datasets\\DATASETS\\' + set_name + '\\' + set_name + '_TRAIN', delimiter=',')
    test = np.loadtxt('.\datasets\\DATASETS\\' + set_name + '\\' + set_name + '_TEST', delimiter=',')

    train_data = train[:,1:]
    train_label = train[:,0:1]
    train = np.concatenate((train_data, train_label),axis = 1)

    
    test_data = test[:,1:]
    test_label = test[:,0:1]
    test = np.concatenate((test_data, test_label),axis = 1)

    return [train, test]

def normalization(label, end):
    num_max, num_min = max(label), min(label)
    nums = len(label)
    for i in range(nums):
        label[i] = end * (label[i] - num_min) / (num_max - num_min)
    return label
