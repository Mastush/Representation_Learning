import torch.nn as nn
from torch.nn.modules import Linear, ReLU, Sigmoid, Softmax
from torch.nn.init import xavier_normal_, zeros_
from torch import from_numpy
import numpy as np
import argparse

import utils


class SimpleNetwork(nn.Module):
    """
    A FC neural network with one hidden layer and one output node
    """
    def __init__(self, d: int, q: int, activation=ReLU, init_f=xavier_normal_, bias=False):
        super(SimpleNetwork, self).__init__()
        self._init_f = init_f
        self._bias = bias

        self._layer1 = Linear(d, q, bias)
        self._activation = activation()
        self._layer2 = Linear(q, 1, bias)
        self._initialize_weights()
        self.to(utils.get_device())

    def forward(self, x):
        x = self._layer1(x)
        x = self._activation(x)
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
    A conv neural network with one hidden layer and one output node
    """
    def __init__(self, input_shape: tuple, c: int, activation=ReLU, init_f=xavier_normal_, bias=False,
                 k: int = 7, stride: int = 1, padding: int = 0, dilation: int = 1):
        super(SimpleConvNetwork, self).__init__()
        self._input_shape = input_shape if len(input_shape) == 4 else (1, *input_shape)
        self._init_f = init_f
        self._bias = bias

        self._conv = nn.Conv2d(self._input_shape[1], c, k, stride, padding, dilation)
        self._activation = activation()
        self.to(utils.get_device())

        self._initialize_weights()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = from_numpy(x).float()
        if len(x.shape) == 3:
            x = x.reshape((1, *x.shape))
        x = self._conv(x)
        x = self._activation(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).mean()
        return x

    def _initialize_weights(self):
        self._init_f(self._conv.weight)
        if self._bias:
            zeros_(self._conv.bias)


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
        x = from_numpy(x).float().to(utils.get_device())
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


class ConvNetwork(nn.Module):
    def __init__(self, input_shape: tuple, c: int, layers: int, kernel_size: int = 3, activation=ReLU,
                 init_f=xavier_normal_, bias: bool = False, pooling: bool = True, auto_pad: bool = False):
        super(ConvNetwork, self).__init__()
        self._init_f = init_f
        self._bias = bias
        self._pooling = pooling
        self._input_shape = input_shape

        self._conv_layers = []
        self._pooling_layers = []
        self._activation_layers = []

        last_c = input_shape[1]
        next_c = c
        for i in range(layers):
            layer = nn.Conv2d(last_c, next_c, kernel_size, padding=kernel_size // 2 if auto_pad else 0)
            activation_layer = activation()
            pooling_layer = nn.MaxPool2d(2)
            self._conv_layers.append(layer)
            last_c = next_c
            next_c *= 2
            if pooling:
                self._pooling_layers.append(pooling_layer)
            self._activation_layers.append(activation_layer)
            self.add_module("Conv layer {}".format(i), layer)
            self.add_module("Activation layer {}".format(i), activation_layer)
            self.add_module("Pooling layer {}".format(i), pooling_layer)
        last_height = utils.get_last_dim_of_conv_network(layers, input_shape[-2], pooling, kernel_size, auto_pad)
        last_width = utils.get_last_dim_of_conv_network(layers, input_shape[-1], pooling, kernel_size, auto_pad)
        self._fc = Linear(last_height * last_width * last_c, 2, bias)

        self._softmax = Softmax()
        self.to(utils.get_device())

    def forward(self, x):
        x = x.reshape((x.shape[0], *self._input_shape[1:])) if len(x.shape) > 1 else x.reshape(self._input_shape)
        x = from_numpy(x).float().to(utils.get_device())
        for i in range(len(self._conv_layers)):
            x = self._conv_layers[i](x)
            if self._pooling:
                x = self._pooling_layers[i](x)
            x = self._activation_layers[i](x)
        x = x.flatten(1)
        x = self._fc(x)
        x = self._softmax(x)
        return x

    def _initialize_weights(self):
        for layer in self._conv_layers:
            self._init_f(layer.weight)
            if self._bias:
                zeros_(layer.bias)


def get_network_for_reps(network_type):
    networks_dict = {"simple": SimpleNetwork, "conv": SimpleConvNetwork}  # TODO: write get network function
    assert network_type.lower() in networks_dict, "Network type {} not supported".format(network_type)
    return networks_dict[network_type]
