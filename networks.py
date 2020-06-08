import torch.nn as nn
from torch.nn.modules import Linear, ReLU, Sigmoid, Softmax
from torch.nn.init import xavier_normal_, zeros_
from torch.nn import BCELoss, MSELoss
from torch.optim import SGD, Adam, Adagrad
from torch import from_numpy
import numpy as np

import utils, evaluation


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

        self._conv = nn.Conv2d(input_shape[0], c, k, stride, padding, dilation)
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
            self._fc_layers.append(Linear(d if i == 0 else q, q, bias=bias))
            self._activation_layers.append(activation())
        self._last_fc = Linear(q, 2)
        self._softmax = Softmax()

        self._initialize_weights()
        self.to(utils.get_device())
        self.float()

    def forward(self, x):
        x = from_numpy(x).float()
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
    def __init__(self, input_shape: tuple, c: int, layers: int, activation=ReLU,
                 init_f=xavier_normal_, bias=False, pooling=True):
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
            self._conv_layers.append(nn.Conv2d(last_c, next_c, 3))
            last_c = next_c
            next_c *= 2
            if pooling:
                self._pooling_layers.append(nn.MaxPool2d(2))
            self._activation_layers.append(activation())
        last_height = utils.get_last_dim_of_conv_network(layers, input_shape[-2], pooling)
        last_width = utils.get_last_dim_of_conv_network(layers, input_shape[-1], pooling)
        self._fc = Linear(last_height * last_width * last_c, 2, bias)

        self._softmax = Softmax()

    def forward(self, x):
        x = x.reshape((x.shape[0], *self._input_shape[1:])) if len(x.shape) > 1 else x.reshape(self._input_shape)
        x = from_numpy(x).float()
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


def train_network(model, x, y, x_test=None, y_test=None, epochs=50, batch_size=64, loss_f=BCELoss,
                  optimizer=Adam, lr=0.001, y_postprocessing=utils.y_to_one_hot, weight_decay: float = 0.0000001):
    print("Started training")

    if y_postprocessing is not None:
        y = y_postprocessing(y)
        y_test = y_postprocessing(y_test)
    y, y_test = y.astype(np.float), y_test.astype(np.float)

    loss_f = loss_f()
    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        for i in range(x.shape[0] // batch_size):
            print("batch {} of {}".format(i + 1, x.shape[0] // batch_size))
            optimizer.zero_grad()
            x_for_network = x[i * batch_size:(i + 1) * batch_size]
            y_for_network = y[i * batch_size:(i + 1) * batch_size]
            pred = model(x_for_network)
            loss = loss_f(pred, from_numpy(y_for_network).float())
            loss.backward()
            optimizer.step()
        performance = evaluation.evaluate_model(model, x, y, pred_postprocessing=utils.softmax_to_one_hot, out_dim=2)
        print("train performance is {}".format(performance))
        print("Finished epoch {}".format(epoch))
        if x_test is not None:
            performance = evaluation.evaluate_model(model, x_test, y_test, pred_postprocessing=utils.softmax_to_one_hot, out_dim=2)
            print("test performance is {}".format(performance))


def get_network(network_type):
    networks_dict = {"simple": SimpleNetwork, "conv": SimpleConvNetwork}  # TODO: write get network function
    assert network_type.lower() in networks_dict, "Network type {} not supported".format(network_type)
    return networks_dict[network_type]