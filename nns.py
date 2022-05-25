import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 10)
        self.fc2 = nn.Linear(hidden_size * 10, hidden_size * 1)
        self.out = nn.Linear(hidden_size * 1, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


class Ou_subNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 15)
        self.fc2 = nn.Linear(hidden_size * 15, hidden_size * 30)
        self.fc3 = nn.Linear(hidden_size * 30, hidden_size * 10)
        self.fc4 = nn.Linear(hidden_size * 10, hidden_size * 5)
        self.out = nn.Linear(hidden_size * 5, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)

        return x

class Lstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1,num_layers=4):
        super(Lstm, self).__init__()
        self.rnn = torch.nn.LSTM(input_size,hidden_size,num_layers)
        self.reg = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x, _ = self.rnn(x.view(len(x), 1, -1))# 单个下划线表示不在意的变量，这里是LSTM网络输出的两个隐藏层状态
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s,b,-1)#使用-1表示第三个维度自动根据原来的shape 和已经定了的s,b来确定
        return x