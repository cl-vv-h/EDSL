import numpy as np
import dis_com
from scipy import stats
import scipy.sparse as sp
import utils

def modefy_simple(data, label, zscore=True):
    [rowSize, columnSize] = data.shape

    res = np.zeros((rowSize * rowSize, 2 * columnSize + 1))
    for i in range(rowSize):
        for j in range(rowSize):
            res[i * rowSize  + j, 0:columnSize] = data[i]
            res[i * rowSize  + j, columnSize:2 * columnSize] = data[j]
            res[i * rowSize  + j, 2 * columnSize] = dis_com.ou_dis(data[i], data[j])
            if label[i] == label[j]:
                res[i * rowSize  + j, 2 * columnSize] /= 20
            else:
                res[i * rowSize  + j, 2 * columnSize] *= 20
    
    if zscore:
        res[:,-1] = stats.zscore(res[:,-1])

    return res

def modefy_simple_sub(data, label, zscore=True):
    [rowSize, columnSize] = data.shape

    res = np.zeros((rowSize * rowSize, columnSize + 1))
    for i in range(rowSize):
        for j in range(rowSize):
            res[i * rowSize  + j, 0:columnSize] = data[i] - data[j]
            res[i * rowSize  + j, columnSize] = sum(res[i * rowSize + j] ** 2) ** 0.5
            if label[i] == label[j]:
                res[i * rowSize  + j, columnSize] /= 10
            else:
                res[i * rowSize  + j, columnSize] *= 10

    res[:,-1] = utils.normalization(res[:,-1],30)
    if zscore:
        res[:,-1] = stats.zscore(res[:,-1])

    return res
    
def modefy_simple_sub_t(data, label, times,zscore=True):
    [rowSize, columnSize] = data.shape

    res = np.zeros((rowSize * rowSize, columnSize + 1))
    for i in range(rowSize):
        for j in range(rowSize):
            res[i * rowSize  + j, 0:columnSize] = data[i] - data[j]
            res[i * rowSize  + j, columnSize] = sum(res[i * rowSize + j] ** 2) ** 0.5
            if label[i] == label[j]:
                res[i * rowSize  + j, columnSize] /= times
            else:
                res[i * rowSize  + j, columnSize] *= times

    res[:,-1] = utils.normalization(res[:,-1],30)
    if zscore:
        res[:,-1] = stats.zscore(res[:,-1])

    return res

def modefy_simple_sub_SBD(data, label, times, zscore=True):
    [rowSize, columnSize] = data.shape

    res = np.zeros((rowSize * rowSize, columnSize + 1))
    for i in range(rowSize):
        for j in range(rowSize):
            res[i * rowSize  + j, 0:columnSize] = data[i] - data[j]
            res[i * rowSize  + j, columnSize] = dis_com.SBD_distance_fft(data[i], data[j])[0]
            if label[i] == label[j]:
                res[i * rowSize  + j, columnSize] /= times
            else:
                res[i * rowSize  + j, columnSize] *= times
    res[:,-1] = utils.normalization(res[:,-1],2)
    #print(res[:,-1])
    if zscore:
        res[:,-1] = stats.zscore(res[:,-1])

    return res

def modefy_simple_sub_gcn(data, label, size, zscore=True):
    [rowSize, columnSize] = data.shape
    edges1, edges2 = [], []

    res = np.zeros((rowSize * rowSize, columnSize + 1))
    for i in range(rowSize):
        for j in range(i+1, rowSize):
            res[i * rowSize  + j, 0:columnSize] = data[i] - data[j]
            res[i * rowSize  + j, columnSize] = sum(res[i * rowSize + j] ** 2) ** 0.5
            if label[i] == label[j]:
                res[i * rowSize  + j, columnSize] /= size
                edges1.append(i)
                edges2.append(j)
            else:
                res[i * rowSize  + j, columnSize] *= size

    if zscore:
        res[:,-1] = stats.zscore(res[:,-1])

    adj = sp.coo_matrix(np.ones(len(edges1)), (edges1, edges2),  shape=(rowSize * rowSize, rowSize * rowSize), dtype = np.float32)

    return res, adj

def modefy_simple_sub_ratio(data, label, method, zscore=True):
    [rowSize, columnSize] = data.shape
    congen = []
    heter = []
    res = np.zeros((rowSize * rowSize, columnSize + 1))
    for i in range(rowSize):
        for j in range(i+1, rowSize):
            res[i * rowSize  + j, 0:columnSize] = data[i] - data[j]
            res[i * rowSize  + j, columnSize] = method(data[i], data[j])
            if label[i] == label[j]:
                if res[i * rowSize  + j, columnSize] > 0.05:
                    congen.append(res[i * rowSize  + j, columnSize])
                #res[i * rowSize  + j, columnSize] /= 20
            else:
                heter.append(res[i * rowSize  + j, columnSize])
                #res[i * rowSize  + j, columnSize] *= 20

    Ymin = min(congen)
    Ymax = max(heter)
    print('Ymax: ', Ymax, 'Ymin: ', Ymin)
    for i in range(rowSize):
        for j in range(i+1, rowSize):
            tmp = res[i * rowSize  + j, columnSize]
            if label[i] == label[j]:
                res[i * rowSize  + j, columnSize] = (tmp * Ymin) ** 0.5
            else:
                res[i * rowSize  + j, columnSize] = (tmp * Ymax) ** 0.5
    res[:,-1] = utils.normalization(res[:,-1],30)

    #print(res[:,-1])
    if zscore:
        res[:,-1] = stats.zscore(res[:,-1])

    return res