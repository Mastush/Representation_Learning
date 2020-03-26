import torch.nn as nn
from torch.nn.modules import Linear, ReLU, Sigmoid
from torch.nn.init import xavier_normal_, zeros_, ones_

import utils


class SimpleNetwork(nn.Module):
    """
    A FC neural network with one hidden layer and one output node
    """
    def __init__(self, d: int, q: int, activation=ReLU, init_f=xavier_normal_, bias=False):
        """
        :param d: Dimension of the input data
        :param q: Number of hidden neurons in layer 1
        :param activation: Activation function
        :param init_f: Initialization function
        """
        super(SimpleNetwork, self).__init__()
        self._init_f = init_f
        self._bias = bias

        self._layer1 = Linear(d, q, bias)
        self._activation1 = activation()
        self._layer2 = Linear(q, 1, bias)
        self._initialize_weights()
        self.to(utils.get_device())

    def forward(self, x):
        x = self._layer1(x)
        x = self._activation1(x)
        x = self._layer2(x)
        return x

    def _initialize_weights(self):
        self._init_f(self._layer1.weight)
        self._init_f(self._layer2.weight)
        if self._bias:
            zeros_(self._layer1.bias)
            zeros_(self._layer2.bias)



