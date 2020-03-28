import torch
import numpy as np


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_tensor_to_ndarray(tensor):
    return np.copy(tensor.data.cpu().numpy())


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def normalize_vectors(mat):
    mags = np.linalg.norm(mat, axis=0)
    return np.divide(mat, mags)
