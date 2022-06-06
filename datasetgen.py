import numpy as np
from train import train
import utils 
import truth_improve

#dataset_name = 'DistalPhalanxOutlineCorrect'
datasets = ['Car', 'Earthquakes', 'Computers', 'CBF', 'DistalPhalanxOutlineCorrect', 'ChlorineConcentration', 'Ham', 'Haptics', 'MedicalImages', 'uWaveGestureLibrary_X']
for dataset_name in datasets:
    #data = utils.loadData('Car')
    data, test = utils.loadinDATASETS(dataset_name)
    #np.random.shuffle(data)
    data = np.concatenate((data, test))
    rowSize = data.shape[0]
    trainSize = min(max(rowSize // 4, 100), 400)
    testSize = rowSize - trainSize
    train_x, train_y = data[:trainSize,0:-1], data[:trainSize,-1]
    test_x, test_y = data[trainSize:,0:-1], data[trainSize:,-1]

    #train_data = truth_improve.modefy_simple_sub_SBD(data[:, :-1],data[:,-1], 1, zscore=False)
    #np.random.shuffle(train_data)
    distances = truth_improve.modefy_simple_sub_t(data[:trainSize, :-1],data[:trainSize,-1], 20, zscore=False)
    print(distances.shape)

    np.save('data/' + dataset_name, distances)