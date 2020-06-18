from abc import ABC, abstractmethod
import torch
import numpy as np

from networks import SimpleNetwork, SimpleConvNetwork
import utils


class BaseRepresentation(ABC):
    """A base class for representations"""
    @abstractmethod
    def __call__(self, x, *args):
        pass

    @abstractmethod
    def get_output_shape(self, x=None):
        pass


class SequentialRepresentation(BaseRepresentation):
    def __init__(self, representations: list):
        self._sub_representations = representations

    def __call__(self, x, *args):
        for rep in self._sub_representations:
            x = rep(x)
        return x

    def get_output_shape(self, x=None):
        return self._sub_representations[-1].get_output_shape(x)


# ---------- Network Representations ---------- #


def get_net_rep(network, input_shape=None) -> list:
    """Returns a list of relevant nn representations"""
    if isinstance(network, SimpleNetwork):
        return [SimpleNetworkGradientRepresentation(network)]
    if isinstance(network, SimpleConvNetwork):
        conv_net_rep = SimpleConvNetworkGradientRepresentation(network, input_shape)
        patched_conv_net_rep = PatchedSimpleConvGradRepresentation(conv_net_rep)  # note potential memory issue of placing the patched rep here
        return [conv_net_rep, patched_conv_net_rep]
    else:
        raise TypeError("Network type {} not supported".format(network.__name__))


class SimpleNetworkGradientRepresentation(BaseRepresentation):
    def __init__(self, model: SimpleNetwork):
        self._model = model.float()

    def _closed_form_call(self, x):
        W = utils.safe_tensor_to_ndarray(self._model._layer1.weight)  # TODO: save these as self.X
        V = utils.safe_tensor_to_ndarray(self._model._layer2.weight)
        activation = self._model._activation
        Wx = np.matmul(W, x)
        y = utils.get_activation_gradient(activation, Wx)
        y = utils.safe_tensor_to_ndarray(y)
        y = np.outer(y, x)
        y = np.multiply(V, y.T).T
        return y

    def __call__(self, x, return_ndarray=True, closed_form=True):
        if closed_form:
            return self._closed_form_call(x)

        self._model.zero_grad()
        self._model.train()
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(utils.get_device())
        y = self._model(x)
        y.backward()
        gradient_matrix = utils.safe_tensor_to_ndarray(self._model._layer1.weight.grad) if \
            return_ndarray else self._model._layer1.weight.grad

        return gradient_matrix

    def get_output_shape(self, x=None):
        return utils.safe_tensor_to_ndarray(self._model._layer1.weight.grad).shape


class SimpleConvNetworkGradientRepresentation(BaseRepresentation):  # TODO: this can be simplified to use closed forms
    def __init__(self, model: SimpleConvNetwork, input_shape=None):
        self._model = model.float()
        self.input_shape = input_shape if len(input_shape) == 4 else (1, *input_shape)

    def get_output_shape(self, x=None):
        weight = utils.safe_tensor_to_ndarray(self._model._conv.weight)
        return weight.shape

    def get_kernel_size(self):
        return self.get_output_shape()[-1]

    def _closed_form_call(self, x):  # TODO: REDO
            return 1

    def __call__(self, x, return_ndarray=True, closed_form=False):
        self._model.zero_grad()
        self._model.train()
        if x.size == utils.shape_to_size(self.input_shape):
            x = np.reshape(x, self.input_shape)
        else:
            if len(x.shape) == 2:
                x = np.reshape(x, (1, 1, *x.shape))
            if len(x.shape) == 3:
                x = np.reshape(x, (1, *x.shape))
            # we try to square the input while keeping the num of channels
            b, c = x.shape[0], x.shape[1]
            channel_size = x.size / (b * c)
            side_size = np.sqrt(channel_size)
            assert int(side_size) == side_size, "Bad input size {}".format(x.shape)
            side_size = int(side_size)
            x = np.reshape(x, (b, c, side_size, side_size))

        if closed_form:
            return self._closed_form_call(x)

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(utils.get_device())
        y = self._model(x)
        y.backward()
        gradient_matrix = utils.safe_tensor_to_ndarray(self._model._conv.weight.grad) if \
            return_ndarray else self._model._conv.weight.grad

        return gradient_matrix


class PatchedSimpleConvGradRepresentation(BaseRepresentation):
    def __init__(self, simple_conv_grad_rep: SimpleConvNetworkGradientRepresentation):
        self._inner_rep = simple_conv_grad_rep
        self._patch_size = simple_conv_grad_rep.get_kernel_size()

    def __call__(self, x, pad: int = 0):
        x = np.reshape(x, self._inner_rep.input_shape[1:])
        if pad > 0:
            x = utils.pad_image(x, pad, pad)
        patches = utils.image_to_patches(x, self._patch_size)
        y = np.zeros((*patches.shape[:2], *self._inner_rep.get_output_shape()))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                y[i, j, :] = self._inner_rep(patches[i, j])
        return y

    def get_output_shape(self, x=None):
        kernel_rad = self._inner_rep.get_kernel_size() // 2
        return (x.shape[0] - 2 * kernel_rad, x.shape[1] - 2 * kernel_rad, x.shape[3], *self._inner_rep.get_output_shape())

# ---------- Linear Separator Representations ---------- #


def get_separator_rep(nn_rep, w, shape=None):
    if isinstance(nn_rep, SimpleNetworkGradientRepresentation):
        return MatrixRepresentation(w, shape)
    if isinstance(nn_rep, PatchedSimpleConvGradRepresentation):
        return ConvSeparatorRepresentation(w)
    else:
        raise TypeError("Network representation type {} not supported".format(nn_rep.__name__))


class MatrixRepresentation(BaseRepresentation):  # TODO: add get imput& output shapes
    def __init__(self, M: np.ndarray, shape=None, mode: int = 1):
        """
        :param M: a matrix or a vector, make sure that the last entry is NOT a bias term!
        :param shape: shape
        :param mode: integer
        """
        self._M = M if shape is None else M.reshape(shape)
        self._call_f = None

    def check_and_fix_shape(self, x):
        x = np.squeeze(x)
        if x.T.shape == self._M.shape:
            return x.T
        elif x.shape == self._M.shape:
            return x
        else:
            raise ValueError("x has shape {} instead of {}".format(x.shape,
                                                                   self._M.shape))

    def __call__(self, x, *args):
        x = self.check_and_fix_shape(x)
        return np.sum(self._M * x, axis=tuple(range(1, x.ndim)))

    def get_output_shape(self, x=None):
        return self._M.shape[0]


class ConvSeparatorRepresentation(BaseRepresentation):
    def __init__(self, separator):
        self._w = separator

    def __call__(self, x, *args):
        assert x[0, 0].size == self._w.size, "x shape is {} and separator shape is {}".format(x.shape, self._w.shape)
        if len(self._w.shape) == 1:
            self._w = np.reshape(self._w, x.shape[2:])
        y = np.zeros(x.shape[:3])
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j, :] = (x[i, j] * self._w).sum(axis=tuple(range(1, len(x[i, j].shape))))
        return np.rollaxis(y, -1, 0)

    def get_output_shape(self, x=None):
        return tuple(np.roll(x.shape[:3], 1))