import numpy as np
from abc import abstractmethod
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

import utils


class RepresentableDataset:  # TODO: allow standartization after each rep?
    def __init__(self, representations: list, standardize_raw: bool = True, standardize_all: bool = True):
        self._representations = representations
        self._standardization_mean = [None] * (len(representations) + 1)
        self._standardization_std = [None] * (len(representations) + 1)
        self._standardize_raw = standardize_raw
        self._standardize_all = standardize_all

    @abstractmethod
    def get_training_examples(self, n: int = None, apply_representations: bool = True):
        """ If n is None, then this should return an entire dataset """
        pass

    def apply_representations_to_data(self, x):
        if self._standardize_raw:
            self._standardize_data(x, 0)
        for i, rep in enumerate(self._representations):
            x = np.asarray([rep(x[j]) for j in range(x.shape[0])])  # TODO: allow vectorized way
            if self._standardize_all:
                self._standardize_data(x, i + 1)
        return x

    def apply_representation_to_example(self, x):
        if self._standardize_raw:
            self._standardize_data(x, 0)
        for i, rep in enumerate(self._representations):
            x = rep(x)
            if self._standardize_all:
                self._standardize_data(x, i + 1)
        return x

    def append_representation(self, rep):
        self._representations.append(rep)
        self._standardization_mean.append(None)
        self._standardization_std.append(None)

    def update_representations(self, representations):
        self._representations = representations
        self._standardization_mean = [None] * (len(representations) + 1)
        self._standardization_std = [None] * (len(representations) + 1)

    def _standardize_data(self, x, phase: int, redo_standardization: bool = False):
        if self._standardization_mean[phase] is None or redo_standardization:
            assert self._standardization_std[phase] is None
            x, mean, std = utils.standardize_data(x)
            self._standardization_mean[phase] = mean
            self._standardization_std[phase] = std
        else:
            assert self._standardization_std[phase] is not None
            x -= self._standardization_mean[phase]
            x /= self._standardization_std[phase]
        return x


class RepresentableVectorDataset(RepresentableDataset):
    def __init__(self, representations: list, normalize: bool = False,
                 standardize_raw: bool = True, standardize_all: bool = True):
        super(RepresentableVectorDataset, self).__init__(representations, standardize_raw, standardize_all)
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
                     dim_reduction: int = None, shuffle: bool = False, y_preprocessing=None):
        x = np.copy(x)
        y = np.copy(y)
        if self._PCA is None and dim_reduction is not None:  # TODO: allow redoing PCA
            self._make_and_fit_pca(x, dim_reduction)
        if shuffle:
            x, y = utils.unison_shuffled_copies(x, y)
        x, y = (x, y) if n is None else (x[:n], y[:n])
        if dim_reduction is not None:
            x = self._apply_pca(x, dim_reduction)
        if self._normalize:
            x = utils.normalize_vectors(x)
        if apply_representations:
            x = self.apply_representations_to_data(x)
        if y_preprocessing is not None:
            y = y_preprocessing(y)
        return x, y


class RepresentableMnist(RepresentableVectorDataset):
    def __init__(self, representations: list, standardize_raw: bool = True, standardize_all: bool = True):
        super(RepresentableMnist, self).__init__(representations, standardize_raw, standardize_all)
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        THRESHOLD = 60000
        self._training_ims = x[:THRESHOLD]
        self._training_labels = y[:THRESHOLD]
        self._test_ims = x[THRESHOLD:]
        self._test_labels = y[THRESHOLD:]
        self._PCA = None

        def mnist_to_binary_truth(truth):
            truth = (truth.astype(np.int) > 4).astype(np.int)
            truth[truth == 0] = -1
            return truth
        self._y_preprocessing = mnist_to_binary_truth

    def get_training_examples(self, n: int = None, apply_representations: bool = True,
                              dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._training_ims, self._training_labels, n, apply_representations,
                                 dim_reduction, shuffle, self._y_preprocessing)

    def get_test_examples(self, n: int = None, apply_representations: bool = True,
                          dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._test_ims, self._test_labels, n, apply_representations,
                                 dim_reduction, shuffle, self._y_preprocessing)

