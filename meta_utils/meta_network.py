"""
Some meta networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.quantize import Function_STE, Function_BWN
from utils.miscellaneous import progress_bar
from utils.quantize import quantized_CNN, quantized_Linear
import utils.global_var as gVar

meta_count = 0

class MetaLSTMFC(nn.Module):

    def __init__(self, hidden_size = 20):
        super(MetaLSTMFC, self).__init__()

        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size=1, hidden_size = hidden_size, num_layers=1)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x, hidden = None):

        if hidden is None:
            x, (hn1, cn1) = self.lstm1(x)
        else:
            x, (hn1, cn1) = self.lstm1(x, (hidden[0], hidden[1]))

        x = self.fc1(x.view(-1, self.hidden_size))

        return x, (hn1, cn1)


class MetaMultiLSTMFC(nn.Module):

    def __init__(self, hidden_size=20, num_lstm=2):
        super(MetaMultiLSTMFC, self).__init__()

        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_lstm)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x, hidden=None):

        if hidden is None:
            x, (hn1, cn1) = self.lstm1(x)
        else:
            x, (hn1, cn1) = self.lstm1(x, (hidden[0], hidden[1]))

        x = self.fc1(x.view(-1, self.hidden_size))

        return x, (hn1, cn1)


class MetaFC(nn.Module):

    def __init__(self, hidden_size = 1500, symmetric_init=False, use_nonlinear=None):
        super(MetaFC, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=hidden_size, bias=False)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        if symmetric_init:
            self.linear1.weight.data.fill_(1.0 / hidden_size)
            self.linear2.weight.data.fill_(1.0)

        self.use_nonlinear = use_nonlinear

    def forward(self, x):

        x = self.linear1(x)
        if self.use_nonlinear is 'relu':
            x = F.relu(x)
        elif self.use_nonlinear is 'tanh':
            x = torch.tanh(x)
        x = self.linear2(x)

        return x


class MetaMultiFC(nn.Module):

    def __init__(self, hidden_size = 10, use_nonlinear=None):
        super(MetaMultiFC, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=hidden_size, bias=False)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        self.use_nonlinear = use_nonlinear

    def forward(self, x):

        x = self.linear1(x)
        if self.use_nonlinear == 'relu':
            x = F.relu(x)
        elif self.use_nonlinear == 'tanh':
            x = torch.tanh(x)
        x = self.linear2(x)
        if self.use_nonlinear == 'relu':
            x = F.relu(x)
        elif self.use_nonlinear == 'tanh':
            x = torch.tanh(x)
        x = self.linear3(x)

        return x


class MetaDesignedMultiFC(nn.Module):

    def __init__(self, hidden_size = 10, num_layers = 4, use_nonlinear='relu'):
        super(MetaDesignedMultiFC, self).__init__()

        self.use_nonlinear = use_nonlinear
        self.network = nn.Sequential()
        # self.linear = dict()
        for layer_idx in range(num_layers):

            in_features = 1 if layer_idx == 0 else hidden_size
            out_features = 1 if layer_idx == (num_layers-1) else hidden_size

            self.network.add_module('Linear%d' %layer_idx, nn.Linear(in_features=in_features, out_features=out_features, bias=False))

            if layer_idx != (num_layers-1):
                if self.use_nonlinear == 'relu':
                    self.network.add_module('ReLU%d' %layer_idx, nn.ReLU())
                elif self.use_nonlinear == 'tanh':
                    self.network.add_module('Tanh%d' %layer_idx, nn.Tanh())
                else:
                    # raise NotImplementedError
                    pass

    def forward(self, x):

        return self.network(x)


class MetaMultiFCBN(nn.Module):

    def __init__(self, hidden_size = 10, use_nonlinear = None):
        super(MetaMultiFCBN, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=hidden_size, bias=False)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)

        self.use_nonlinear = use_nonlinear

    def forward(self, x):

        x = self.linear1(x)
        x = self.bn1(x)
        if self.use_nonlinear == 'relu':
            x = F.relu(x)
        elif self.use_nonlinear == 'tanh':
            x = torch.tanh(x)
        x = self.linear2(x)
        x = self.bn2(x)
        if self.use_nonlinear == 'relu':
            x = F.relu(x)
        elif self.use_nonlinear == 'tanh':
            x = torch.tanh(x)
        x = self.linear3(x)

        return x


class MetaSimple(nn.Module):
    """
    A simple Meta model just multiplies a factor to the input gradient
    """
    def __init__(self):
        super(MetaSimple, self).__init__()

        self.alpha = nn.Parameter(torch.ones([1]))

    def forward(self, x):

        return self.alpha * x


def update_parameters(net, lr):
    for param in net.parameters():
        param.data.add_(-lr * param.grad.data)


def test(net, quantized_type, test_loader, use_cuda = True):

    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs, quantized_type)

        _, predicted = torch.max(outputs.data, dim=1)
        correct += predicted.eq(targets.data).cpu().sum().item()
        total += targets.size(0)
        progress_bar(batch_idx, len(test_loader), "Test Acc: %.3f%%" % (100.0 * correct / total))

    return 100.0 * correct / total


if __name__ == '__main__':

    net = MetaDesignedMultiFC()

    torch.save(
        {
            'model': net,
            'hidden_size': 100,
            'nonlinear': 'None'
        }, './Results/meta_net.pkl'
    )

    meta_pack = torch.load('./Results/meta_net.pkl')

    retrieve_net = meta_pack['model']
    inputs = torch.rand([10, 1])
    outputs = retrieve_net(inputs)


