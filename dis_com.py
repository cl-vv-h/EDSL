import numpy as np
from scipy.fftpack import ifft, fft

def SBD_distance_fft(point1, point2):
    i = np.log2(2 * len(point1) - 1)
    i = np.floor(i) + 1
    i = int(i)
    t = 2 ** i - 1
    fft_p1 = fft(point1, n=t)
    fft_p2 = fft(point2, n=t)
    fft_p1p2 = np.empty((len(fft_p1),), dtype=np.complex128, order='C')
    for i in range(0, len(fft_p1)):
        fft_p1p2[i] = fft_p1[i] * np.conj(fft_p2[i])

    ifft_p1p2 = ifft(fft_p1p2)
    t = np.max(ifft_p1p2.real) / (len_point(point1) * len_point(point2))
    maxcor = 1 - t

    alligned_point = [0 for m in range(len(point1))]
    fit_i = np.argmax(ifft_p1p2.real) - len(point1)
    fit_i = int(fit_i)
    # print(fit_i)
    zeros = np.zeros(np.array(point2).shape)
    if not (point2 == zeros).all():
        if fit_i > 0:
            for m in range(0, len(point1)):
                if m < fit_i:
                    alligned_point[m] = 0
                else:
                    alligned_point[m] = point1[m - fit_i]

        else:
            for m in range(0, len(point1)):
                if m < len(point1) + fit_i:
                    alligned_point[m] = point1[m - fit_i]
                else:
                    alligned_point[m] = 0
    else:
        alligned_point = point1
    return [maxcor, alligned_point]

def len_point(point):
    return np.linalg.norm(point)

def n2norm(data, x):
    '''
    tmp = data
    m = tmp.shape[0]
    for i in range(m):
        tmp[i,:] = tmp[i,:] - x
    return np.linalg.norm(data, axis = 1)
    '''
    rowSize = data.shape[0]
    # 计算训练样本和测试样本的差值
    diff = np.tile(x, (rowSize, 1)) - data
    # 计算差值的平方和
    sqrDiff = diff ** 2
    sqrDiffSum = sqrDiff.sum(axis=1)
    # 计算距离
    distances = sqrDiffSum ** 0.5

    return distances

def SBD(data, x):
    rowSize = data.shape[0]
    distances = np.zeros((rowSize, ))
    for i in range(rowSize):
        distances[i] = SBD_distance_fft(data[i], x)[0]

    return distances

def cos(data, x):

    rowSize = data.shape[0]
    product1 = np.tile(x, (rowSize,1)) * data
    sqr = product1.sum(axis=1) ** 0.5
    product2 = sum(x * x) * (data ** 2).sum(axis=1)
    sqr2 = product2 ** 0.5
    distances = sqr / sqr2

    return distances

def cos_dis(x, y):

    return sum(x * y)/((sum(x * x) * sum(y * y)) ** 0.5)

def ou_dis(x, y):
    return sum((x - y) ** 2) ** 0.5
