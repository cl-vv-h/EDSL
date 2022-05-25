import numpy as np
import utils

def predict(x, train_x, train_y, k, disfunc):
    
    norms = disfunc(train_x, x)
    kindex = np.argsort(norms)[:k]

    klabels = []
    for i in kindex:
        klabels.append(train_y[i])
        
    res = max(klabels, key=klabels.count)

    return res

def predict_net(x, train_x, train_y, k, model):
    print(x.shape)
    norms = dis_net(train_x, x, model)
    kindex = np.argsort(norms)[:k]

    klabels = []
    for i in kindex:
        klabels.append(train_y[i])
        
    res = max(klabels, key=klabels.count)

    return res

def predict_net_sub(x, train_x, train_y, k, model):
    
    norms = dis_net_sub(train_x, x, model)
    kindex = np.argsort(norms)[0:k]

    klabels = []
    for i in kindex:
        klabels.append(train_y[i])
        
    res = max(klabels, key=klabels.count)

    return res

def predict_one2one(x, train_x, train_y, k, disfunc):
    rowSize = train_x.shape[0]
    distances = np.zeros((rowSize,))
    for i in range(rowSize):
        distances[i] = disfunc(x, train_x[i])[0]
    kindex = np.argsort(distances)[-k:]

    klabels = []
    for i in kindex:
        klabels.append(train_y[i])
    
    res = max(klabels, key=klabels.count)
    
    return res

def dis_net(data, x, model):
    [m, n] = data.shape
    j = x.shape[0]
    data_train = np.zeros((m, n + j))
    for i in range(m):
        data_train[i,0:n] = data[i]
        data_train[i,n:n+j] = x
    res = np.zeros((m,))
    for i in range(m):
        res[i] = model(utils.np2tensor(data_train[i]).view(1,-1).cuda())

    return res

def dis_net_sub(data, x, model):
    [m, n] = data.shape
    data_train = np.zeros(data.shape)
    for i in range(m):
        data_train[i,0:n] = data[i] - x
    res = np.zeros((m,))

    for i in range(m):
        res[i] = model(utils.np2tensor(data_train[i]).view(1,-1).cuda())

    return res
    
    


'''
train = np.zeros((5,5))
for i in range(5):
    for j in range(4):
        train[i, j] = j + i
for i in range(5):
    if i <= 2:
        train[i, 4] = 0
    else:
        train[i, 4] = 999

x = np.ones((4,))
for i in range(4):
    x[i] = 1

print(predict(x, train, 2))
'''