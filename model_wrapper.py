from abc import ABC, abstractmethod
from sklearn.linear_model import SGDClassifier

import utils


class ModelWrapper(ABC):
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def __call__(self, x):
        pass


class SVMWrapper(ModelWrapper):
    def __init__(self, model: SGDClassifier):
        super(SVMWrapper, self).__init__(model)

    def __call__(self, x):
        if len(x.shape) > 2:
            x = utils.flatten_data(x)
        return self._model.predict(x)
