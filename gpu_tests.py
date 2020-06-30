import torch
import torch.nn as nn
from torch.nn.modules import Linear, ReLU, Sigmoid, Softmax
from torch.nn.init import xavier_normal_, zeros_
from torch import from_numpy
import numpy as np

import utils




class FCNetwork(nn.Module):
    """
    A FC neural network with one hidden layer and one output node
    """
    def __init__(self, d: int, q: int, layers: int, activation=ReLU, init_f=xavier_normal_, bias=False):
        """
        :param d: Dimension of the input data
        :param q: Number of hidden neurons in layer 1
        :param layers: Number of Layers
        :param activation: Activation function
        :param init_f: Initialization function
        """
        super(FCNetwork, self).__init__()
        self._init_f = init_f
        self._bias = bias

        self._fc_layers = []
        self._activation_layers = []

        for i in range(layers):
            layer = Linear(d if i == 0 else q, q, bias=bias)
            activation_layer = activation()
            self._fc_layers.append(layer)
            self._activation_layers.append(activation_layer)
            self.add_module("Dense layer {}".format(i), layer)
            self.add_module("Activation layer {}".format(i), activation_layer)
        self._last_fc = Linear(q, 2)
        self._softmax = Softmax()

        self._initialize_weights()
        self.float()
        self.to(utils.get_device())

    def forward(self, x):
        print("Device is {}".format(utils.get_device()))
        x = from_numpy(x).float().to(utils.get_device())
        print("Is x cuda? {}".format(x.is_cuda))
        for i in range(len(self._fc_layers)):
            x = self._fc_layers[i](x)
            x = self._activation_layers[i](x)
        x = self._last_fc(x)
        x = self._softmax(x)
        return x

    def _initialize_weights(self):
        for layer in self._fc_layers:
            self._init_f(layer.weight)
            if self._bias:
                zeros_(layer.bias)




device = utils.get_device()
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


model = FCNetwork(2, 2, 2)
print("Is the model cuda? {}".format(next(model.parameters()).is_cuda))
x = np.random.rand(5, 2)
for i in range(x.shape[0]):
    y = model(x)