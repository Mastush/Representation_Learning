from abc import ABC, abstractmethod
from torch.nn import BCELoss, Module
import torch
import numpy as np

from networks import SimpleNetwork, SimpleConvNetwork
import utils


class BaseRepresentation(ABC):
    @abstractmethod
    def __call__(self, x, *args):
        pass


class SequentialRepresentation(BaseRepresentation):
    def __init__(self, representations: list):
        self._sub_representations = representations

    def __call__(self, x, *args):
        for rep in self._sub_representations:
            x = rep(x)
        return x


class SimpleNetworkGradientRepresentation(BaseRepresentation):  # TODO: this can be simplified to use closed forms
    def __init__(self, model: SimpleNetwork):
        self._model = model.float()

    def __call__(self, x, return_ndarray=True):
        self._model.zero_grad()
        self._model.train()
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(utils.get_device())
        y = self._model(x)
        y.backward()
        gradient_matrix = utils.safe_tensor_to_ndarray(self._model._layer1.weight.grad) if \
            return_ndarray else self._model._layer1.weight.grad

        return gradient_matrix


class SimpleConvNetworkGradientRepresentation(BaseRepresentation):  # TODO: this can be simplified to use closed forms
    def __init__(self, model: SimpleConvNetwork):
        self._model = model.float()

    def __call__(self, x, return_ndarray=True):
        self._model.zero_grad()
        self._model.train()
        x = np.reshape(x, (1, *self._model.input_shape))

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(utils.get_device())
        y = self._model(x)
        y.backward()
        gradient_matrix = utils.safe_tensor_to_ndarray(self._model._conv.weight.grad) if \
            return_ndarray else self._model._conv.weight.grad

        return np.squeeze(gradient_matrix)


class MatrixRepresentation(BaseRepresentation):  # TODO: add get imput& output shapes
    def __init__(self, M: np.ndarray, shape=None, mode: int = 1):
        """
        :param M: a matrix or a vector, make sure that the last entry is NOT a bias term!
        :param shape: shape
        :param mode: integer
        """
        self._M = M if shape is None else M.reshape(shape)
        self._call_functions_dict = {
                                     1: self.call_mode_1,
                                     2: self.call_mode_2,
                                     3: self.call_mode_3,
                                     4: self.call_mode_4
                                     }
        self._call_f = None
        self.set_call_mode(mode)

    def set_call_mode(self, mode: int):
        self._call_f = self._call_functions_dict[mode]

    def check_and_fix_shape(self, x):
        x = np.squeeze(x)
        if x.T.shape == self._M.shape:
            return x.T
        elif x.shape == self._M.shape:
            return x
        else:
            raise ValueError("x has shape {} instead of {}".format(x.shape,
                                                                   self._M.shape))

    def call_mode_1(self, x):
        """
        If x is in R^(q x d), the output is in R^q
        """
        # return np.sum(x, axis=tuple(range(1, x.ndim)))
        return np.sum(self._M * x, axis=tuple(range(1, x.ndim)))
        # return np.sum(self._M * x, axis=tuple(range(1, x.ndim)))[:-1]

    def call_mode_2(self, x):
        """
        If x is in R^(q x d), the output is in R^(q x q)
        """
        return np.matmul(self._M, x.T)

    def call_mode_3(self, x):
        """
        If x is in R^(q x d), the output is in R^(q x q)
        """
        return np.matmul(self._M.T, x)

    def call_mode_4(self, x):
        """
        If x is in R^(q x d), the output is in R^(d x d)
        """
        x_1 = self.call_mode_1(x)
        return np.outer(x_1, x_1)

    def __call__(self, x, *args):
        x = self.check_and_fix_shape(x)
        return self._call_f(x)  # TODO: normalize somehow?
        # new_x = self._call_f(x)
        # return new_x / new_x.size
