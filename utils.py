import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


EPSILON = 0.0000001


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_tensor_to_ndarray(tensor: torch.Tensor):
    return np.copy(tensor.data.cpu().numpy())


def unison_shuffled_copies(a: np.ndarray, b: np.ndarray):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def normalize_vectors(mat: np.ndarray):
    mat = flatten_data(mat)
    mags = np.linalg.norm(mat, axis=0) + EPSILON
    return np.divide(mat, mags)


def flatten_data(data: np.ndarray):
    last_dim = 1
    for i in range(1, len(data.shape)):
        last_dim *= data.shape[i]
    return np.reshape(data, (data.shape[0], last_dim))


def standardize_data(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler.transform(x), scaler.mean_, scaler.scale_