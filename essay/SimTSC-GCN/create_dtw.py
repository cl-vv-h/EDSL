from dis import dis
from errno import EDEADLOCK
import os
import argparse
from src.utils import dataset_name
import numpy as np

from src.utils import read_X, read_Y

from dtw import dtw
manhattan_distance = lambda x, y: np.abs(x - y)

dataset_dir = './datasets/UCRArchive_2018'
output_dir = './tmp'

def argsparser():
    parser = argparse.ArgumentParser("SimTSC dtw creator")
    parser.add_argument('--dataset', help='Dataset name', default=dataset_name)

    return parser

def get_dtw(X):
    X = X.copy(order='C').astype(np.float64)

    X[np.isnan(X)] = 0
    distances = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    for i in range(len(X)):
        for j in range(len(X)):
            data = X[i]
            query = X[j]
            #distances[i][j] = dtw.query(data, query, r=min(len(data)-1, len(query)-1, 100))['value']
            distances[i][j], cost_matrix, acc_cost_matrix, path = dtw(data, query, dist=manhattan_distance)
    return distances

def get_dtw_C(X,Y):
    X = X.copy(order='C').astype(np.float64)

    X[np.isnan(X)] = 0
    distances = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    Min, Max = 200,-100
    for i in range(len(X)):
        for j in range(len(X)):
            data = X[i]
            query = X[j]
            #distances[i][j] = dtw.query(data, query, r=min(len(data)-1, len(query)-1, 100))['value']
            distances[i][j], cost_matrix, acc_cost_matrix, path = dtw(data, query, dist=manhattan_distance)
            if Y[i] == Y[j]:
                if distances[i][j]<Min:
                    Min = distances[i][j]
            else:
                if distances[i][j]>Max:
                    Max = distances[i][j]
    if Min < 0.5:
        Min = 0.5
    for i in range(len(X)):
        for j in range(len(X)):
            if Y[i] == Y[j]:
                distances[i][j] = (distances[i][j]*Min) ** 0.5
            else:
                distances[i][j] = (distances[i][j]*Max) ** 0.5
    return distances


if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    result_dir = os.path.join(output_dir, 'ucr_datasets_dtw')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    X = read_X(dataset_dir, args.dataset)
    Y = read_Y(dataset_dir, args.dataset)

    #dtw_arr = get_dtw_C(X,Y)
    dtw_arr = get_dtw(X)
    np.save(os.path.join(result_dir, args.dataset), dtw_arr)
