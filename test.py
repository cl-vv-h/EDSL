import utils
import numpy as np

min_l, max_l = 11110,0
min_n, max_n = 11110,0
min_c, max_c = 10000,0
datasets = ['Car', 'Earthquakes', 'Computers', 'CBF', 'DistalPhalanxOutlineCorrect', 'ChlorineConcentration', 'Ham', 'Haptics', 'MedicalImages', 'uWaveGestureLibrary_X']
for dataset in datasets:
    data, test = utils.loadinDATASETS(dataset)
    data = np.concatenate((data, test),axis = 0)
    print(data.shape)

    tmp = []
    [tmp_n, tmp_l] = data.shape

    for i in range(tmp_n):
        if not data[i,-1] in tmp:
            tmp.append(data[i,-1])

    if tmp_l < min_l:
        min_l = tmp_l
    if tmp_l > max_l:
        max_l = tmp_l
    if tmp_n < min_n:
        min_n = tmp_n
    if tmp_n > max_n:
        max_n = tmp_n
    if len(tmp) > max_c:
        max_c = len(tmp)
    if len(tmp) < min_c:
        min_c = len(tmp)

print('min_length: ' + str(min_l), 'max_length: ' + str(max_l))
print('min_number: ' + str(min_n), 'max_number: ' + str(max_n))
print('min_class: ' + str(min_c), 'max_class: ' + str(max_c))
    