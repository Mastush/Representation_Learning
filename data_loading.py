import numpy as np
from abc import abstractmethod
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import utils
from representation import SequentialRepresentation


class RepresentableDataset:
    """This is a base class for a dataset which will go through a chain of representations"""
    def __init__(self, representations: list, standardize_raw: bool = True, standardize_reps: bool = True,
                 normalize_raw: bool = False, normalize_reps: bool = False):
        self._representations = representations
        self._standardization_mean = [None] * (len(representations) + 1)
        self._standardization_std = [None] * (len(representations) + 1)
        self._standardize_raw = standardize_raw
        self._standardize_reps = standardize_reps
        self._normalize_raw = normalize_raw
        self._normalize_reps = normalize_reps

    @abstractmethod
    def get_training_examples(self, n: int = None, apply_representations: bool = True):
        """ If n is None, then this should return an entire dataset """
        pass

    @abstractmethod
    def get_batch_generators(self, batch_size):
        pass

    def apply_representations_to_data(self, x):
        """Iteratively applies standardization and representations to the data"""
        if self._standardize_raw:
            self._standardize_data(x, 0)
        if self._normalize_raw:
            x = utils.normalize_vectors(x)
        for i, rep in enumerate(self._representations):
            x = np.asarray([rep(x[j]) for j in range(x.shape[0])])  # TODO: allow vectorized way
            if self._standardize_reps:
                self._standardize_data(x, i + 1)
            if self._normalize_reps:
                x = utils.normalize_vectors(x)
        return x

    def append_representation(self, rep):
        self._representations.append(rep)
        self._standardization_mean.append(None)
        self._standardization_std.append(None)

    def remove_representations(self, n):
        self._representations = self._representations[:-n]

    def update_representations(self, representations):
        self._representations = representations
        self._standardization_mean = [None] * (len(representations) + 1)
        self._standardization_std = [None] * (len(representations) + 1)

    def get_representations(self, as_sequential: bool = False):
        return SequentialRepresentation(self._representations) if as_sequential else self._representations

    def get_last_representation(self):
        return self._representations[-1]

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

    def set_standardize_raw(self, true_or_false: bool):
        self._standardize_raw = true_or_false

    def set_standardize_reps(self, true_or_false: bool):
        self._standardize_reps = true_or_false

    @abstractmethod
    def get_raw_input_shape(self, conv=False):
        pass


class RepresentableVectorDataset(RepresentableDataset):
    """A representable dataset whose data are vectors"""
    def __init__(self, representations: list, n_train: int, n_test: int, standardize_raw: bool = True,
                 standardize_reps: bool = True, normalize_raw: bool = False, normalize_reps: bool = False):
        super(RepresentableVectorDataset, self).__init__(representations, standardize_raw, standardize_reps,
                                                         normalize_raw, normalize_reps)
        self._PCA = None
        self._n_train = n_train
        self._n_test = n_test

    def _make_and_fit_pca(self, x, d):
        if d is not None and self._PCA is None:
            self._PCA = PCA(d)
            self._PCA.fit(utils.flatten_data(x), None)

    def _apply_pca(self, x, d: int=None):
        return x if x.shape[-1] == d else self._PCA.transform(utils.flatten_data(x))

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
        x = np.copy(x)  #TODO: check if I can remove these copies
        y = np.copy(y)
        if self._PCA is None and dim_reduction is not None:  # TODO: allow redoing PCA
            self._make_and_fit_pca(x, dim_reduction)
        if shuffle:
            x, y = utils.unison_shuffled_copies(x, y)
        x, y = (x, y) if n is None else (x[:n], y[:n])
        if dim_reduction is not None:
            x = self._apply_pca(x, dim_reduction)
        if apply_representations:
            x = self.apply_representations_to_data(x)
        if y_preprocessing is not None:
            y = y_preprocessing(y)
        return x, y

    @abstractmethod
    def get_batch_generators(self, batch_size):
        pass


class RepresentableMnist(RepresentableVectorDataset):
    def __init__(self, representations: list,
                 standardize_raw: bool = True, standardize_reps: bool = True,
                 normalize_raw: bool = False, normalize_reps: bool = False):
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        super(RepresentableMnist, self).__init__(representations, 60000, 10000, standardize_raw, standardize_reps,
                                                 normalize_raw, normalize_reps)
        THRESHOLD = 60000
        self._training_ims = x[:THRESHOLD]
        self._training_labels = y[:THRESHOLD]
        self._test_ims = x[THRESHOLD:]
        self._test_labels = y[THRESHOLD:]
        self._PCA = None

        self._y_preprocessing = utils.get_multiclass_to_binary_truth_f(10)

    def get_training_examples(self, n: int = None, apply_representations: bool = True,
                              dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._training_ims, self._training_labels, n, apply_representations,
                                 dim_reduction, shuffle, self._y_preprocessing)

    def get_test_examples(self, n: int = None, apply_representations: bool = True,
                          dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._test_ims, self._test_labels, n, apply_representations,
                                 dim_reduction, shuffle, self._y_preprocessing)

    def get_batch_generators(self, batch_size):
        def train_batch_generator():
            i = 0
            batch_x, batch_y = None, None
            while i < self._n_train:
                batch_x, batch_y = self.get_examples(self._training_ims[i:i + batch_size],
                                                     self._training_labels[i:i + batch_size])
                i += batch_size
            yield batch_x, batch_y

        def test_batch_generator():
            i = 0
            batch_x, batch_y = None, None
            while i < self._n_test:
                batch_x, batch_y = self.get_examples(self._test_ims[i:i + batch_size],
                                                     self._test_labels[i:i + batch_size])
                i += batch_size
            yield batch_x, batch_y

        return train_batch_generator(), test_batch_generator()

    @abstractmethod
    def get_raw_input_shape(self, conv=False):
        return (1, 1, 28, 28) if conv else 784


class RepresentableCIFAR10(RepresentableVectorDataset):
    def __init__(self, representations: list,
                 standardize_raw: bool = True, standardize_reps: bool = True,
                 normalize_raw: bool = False, normalize_reps: bool = False):
        super(RepresentableCIFAR10, self).__init__(representations, 50000, 10000, standardize_raw, standardize_reps,
                                                 normalize_raw, normalize_reps)

        train_set = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
        self._training_ims, self._training_labels = utils.get_all_data_from_pytorch_dataset(train_set)

        test_set = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
        self._test_ims, self._test_labels = utils.get_all_data_from_pytorch_dataset(test_set)

        self._PCA = None

        self._y_preprocessing = utils.get_multiclass_to_binary_truth_f(10)

    def get_training_examples(self, n: int = None, apply_representations: bool = True,
                              dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._training_ims, self._training_labels, n, apply_representations,
                                 dim_reduction, shuffle, self._y_preprocessing)

    def get_test_examples(self, n: int = None, apply_representations: bool = True,
                          dim_reduction: int = None, shuffle: bool = False):
        return self.get_examples(self._test_ims, self._test_labels, n, apply_representations,
                                 dim_reduction, shuffle, self._y_preprocessing)

    def get_batch_generators(self, batch_size):
        def train_batch_generator():
            i = 0
            batch_x, batch_y = None, None
            while i < self._n_train:
                batch_x, batch_y = self.get_examples(self._training_ims[i:i + batch_size],
                                                     self._training_labels[i:i + batch_size])
                i += batch_size
            yield batch_x, batch_y

        def test_batch_generator():
            i = 0
            batch_x, batch_y = None, None
            while i < self._n_test:
                batch_x, batch_y = self.get_examples(self._test_ims[i:i + batch_size],
                                                     self._test_labels[i:i + batch_size])
                i += batch_size
            yield batch_x, batch_y

        return train_batch_generator(), test_batch_generator()

    @abstractmethod
    def get_raw_input_shape(self, conv=False):
        return (1, 3, 32, 32) if conv else 3072


def get_dataset(dataset_str, normalize_raw, normalize_reps):
    if dataset_str.lower() == 'mnist':
        return RepresentableMnist([], False, False, normalize_raw, normalize_reps)
    elif dataset_str.lower() == 'cifar':
        return RepresentableCIFAR10([], False, False, normalize_raw, normalize_reps)
    else:
        raise ValueError("Dataset {} not supported".format(dataset_str))