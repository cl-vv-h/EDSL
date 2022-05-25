import numpy as np
from train import train
import utils 
import truth_improve


#data = utils.loadData('Car')
data, test = utils.loadinDATASETS('Computers')
np.random.shuffle(data)
print(data.shape)
rowSize = data.shape[0]
trainSize = int(rowSize / 4)
testSize = rowSize - trainSize
train_x, train_y = data[:trainSize,0:-1], data[:trainSize,-1]
test_x, test_y = data[trainSize:,0:-1], data[trainSize:,-1]

train_data = truth_improve.modefy_simple_sub(data[:trainSize, :-1],data[:trainSize,-1], zscore=False)
np.random.shuffle(train_data)
print(train_data.shape)

np.savetxt('trainData.csv', train_data, delimiter=',')
