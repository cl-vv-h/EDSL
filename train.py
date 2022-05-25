from knn import predict
import nns
import torch
import numpy as np
import utils
import dis_com

def train(x, y, netType, hidden_size):
    print('model trainning...')
    input_size = x.shape[1]
    x_ = utils.np2tensor(x)
    y_ = utils.np2tensor(y)

    y_ = y_.reshape((y_.shape[0], 1))
    net = netType(input_size, hidden_size, 1)
    #print(net)

    x_ = x_.cuda()
    y_ = y_.cuda()
    net = net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr = 0.0005)
    loss_F = torch.nn.MSELoss()

    pre = 9999

    for t in range(3500):
        prediction = net(x_)

        loss = loss_F(prediction, y_)
        if pre-loss<0.005:
            break
        #print(loss)
        pre = loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    return net

def data_build(data):
    [m, n] = data.shape
    res = np.zeros((m * m, 2 * n + 1))
    for i in range(m):
        for j in range(m):
            res[i * m  + j, 0:n] = data[i]
            res[i * m  + j, n:2 * n] = data[j]
            res[i * m  + j, 2 * n] = dis_com.ou_dis(data[i], data[j])
    
    return res