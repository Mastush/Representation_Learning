from abc import ABC, abstractmethod
from torch.nn import BCELoss, Module
import torch

from networks import SimpleNetwork
import utils


class Representation(ABC):
    @abstractmethod
    def __call__(self, x, *args):
        pass


class SimpleNetworkGradientRepresentation(Representation):
    def __init__(self,
                 model: SimpleNetwork):
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
