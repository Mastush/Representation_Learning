import torch.nn as nn
from torch.nn.modules import Linear, ReLU, Sigmoid
from torch.nn.init import xavier_normal_, zeros_


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


class SimpleConvNetwork(nn.Module):
    """
    A FC neural network with one hidden layer and one output node
    """
    def __init__(self, k: int, c: int, input_shape: tuple, activation=ReLU, init_f=xavier_normal_, bias=False,
                 stride: int = 1, padding: int = 0, dilation: int = 1):
        super(SimpleConvNetwork, self).__init__()
        self.input_shape = input_shape
        self._init_f = init_f
        self._bias = bias

        self._conv = nn.Conv2d(input_shape[0], c, k, stride, padding, dilation)
        conv_output_shape = utils.conv_output_shape(input_shape[1], input_shape[2], c, k, stride, padding, dilation)
        self._activation = activation()
        self._fc = Linear(utils.shape_to_size(conv_output_shape), 1, bias)
        self._initialize_weights()
        self.to(utils.get_device())

    def forward(self, x):
        x = self._conv(x)
        x = self._activation(x).flatten()
        x = self._fc(x)
        return x

    def _initialize_weights(self):
        self._init_f(self._conv.weight)
        self._init_f(self._fc.weight)
        if self._bias:
            zeros_(self._conv.bias)
            zeros_(self._fc.bias)
