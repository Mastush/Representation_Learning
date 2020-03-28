import numpy as np
from abc import abstractmethod
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

import utils


class RepresentableDataset:
    def __init__(self, representations: list):
        self._representations = representations

    @abstractmethod
    def get_training_examples(self, n: int = None, apply_representations: bool = True):
        """ If n is None, then this should return an entire dataset """
        pass

    def apply_representations(self, x):
        for rep in self._representations:
            x = rep(x)
        return x


class RepresentableVectorDataset(RepresentableDataset):
    def __init__(self, representations: list):
        super(RepresentableVectorDataset, self).__init__(representations)
        self._PCA = None

    def _apply_pca(self, x, d: int=None):
        if d is not None and self._PCA is None:
            self._PCA = PCA(d)
            self._PCA.fit(x, None)
        return self._PCA.transform(x)

    @abstractmethod
    def get_training_examples(self, n: int = None, apply_representations: bool = True,
                              dim_reduction: int = None, shuffle: bool = False):
        pass

    @abstractmethod
    def get_test_examples(self, n: int = None, apply_representations: bool = True,
                          dim_reduction: int = None, shuffle: bool = False):
        pass

    def get_examples(self, x, y, n: int = None, apply_representations: bool = True,
                     dim_reduction: int = None, shuffle: bool = False):
        if dim_reduction is not None:
            x = self._apply_pca(x, dim_reduction)
        else:
            x = np.copy(x)
        y = np.copy(y)
        if shuffle:
            x, y = utils.unison_shuffled_copies(x, y)
        if apply_representations:
            x = self.apply_representations(x)
        return x, y if n is None else x[:n], y[:n]


class RepresentableMnist(RepresentableVectorDataset):
    def __init__(self, representations: list):
        super(RepresentableMnist, self).__init__(representations)
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        self._training_ims = x[:60000]
        self._training_labels = y[:60000]
        self._test_ims = x[60000:]
        self._test_labels = y[:60000:]
        self._PCA = None

    def get_training_examples(self, n: int = None, apply_representations: bool = True,
                              dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._training_ims, self._training_labels, n, apply_representations,
                                 dim_reduction, shuffle)

    def get_test_examples(self, n: int = None, apply_representations: bool = True,
                          dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._test_ims, self._test_labels, n, apply_representations,
                                 dim_reduction, shuffle)

