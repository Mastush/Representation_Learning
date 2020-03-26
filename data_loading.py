import numpy as np
from abc import abstractmethod


class RepresentableDataset:
    def __init__(self, representations: list):
        self._representations = representations

    @abstractmethod
    def get_training_examples(self, n=None, apply_representations=True):
        """ If n is None, then this should return an entire dataset """
        pass

    def apply_representations_to_example(self, example):
        for rep in self._representations:
            example = rep(example)
        return example


class RepresentableMnist(RepresentableDataset):
    def __init__(self, train_path: str, test_path: str, representations: list):
        super(RepresentableMnist, self).__init__(representations)
        self._train_path = train_path
        self._test_path = test_path

    def get_training_examples(self, n=None, apply_representations=True):
        pass
