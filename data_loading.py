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
    def __init__(self, representations: list, normalize: bool=True):
        super(RepresentableVectorDataset, self).__init__(representations)
        self._PCA = None
        self._normalize = normalize

    def _make_and_fit_pca(self, x, d):
        if d is not None and self._PCA is None:
            self._PCA = PCA(d)
            self._PCA.fit(x, None)

    def _apply_pca(self, x, d: int=None):
        return x if x.shape[-1] == d else self._PCA.transform(x)

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
        if self._PCA is None and dim_reduction is not None:
            self._make_and_fit_pca(x, dim_reduction)
        if shuffle:
            x, y = utils.unison_shuffled_copies(x, y)
        x, y = (x, y) if n is None else (x[:n], y[:n])
        if dim_reduction is not None:
            x = self._apply_pca(x, dim_reduction)
        else:
            x = np.copy(x)
        y = np.copy(y)
        if self._normalize:
            x = utils.normalize_vectors(x)
        if apply_representations:
            x = np.asarray([self.apply_representations(x[i]) for i in range(x.shape[0])])
        return x, y


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

