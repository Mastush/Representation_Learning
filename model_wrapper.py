from abc import ABC, abstractmethod
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator

import utils


class ModelWrapper(ABC):
    """
    A class for easy usage of predictors. Allows predictors to be called as functions
    """
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def __call__(self, x):
        pass


class SVMWrapper(ModelWrapper):
    def __init__(self, model: BaseEstimator):  # TODO: change type?
        super(SVMWrapper, self).__init__(model)

    def __call__(self, x):
        if len(x.shape) > 2:
            x = utils.flatten_data(x)
        return self._model.predict(x)

    def get_w(self):
        return self._model.coef_
