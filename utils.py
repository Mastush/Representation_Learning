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
    mean, std = x.mean(axis=0), x.std(0)
    x -= mean
    x /= (std + EPSILON)
    return x, mean, (std + EPSILON)


def conv_output_shape(h, w, channels, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 ) / stride) + 1)
    w = floor(((w + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 ) / stride) + 1)
    return h, w, channels


def add_ones_column(matrix: np.ndarray):
    assert len(matrix.shape) == 2, "Entry must be a matrix!"
    ones_col = np.ones((matrix.shape[0], 1))
    return np.hstack((matrix, ones_col))


def shape_to_size(shape):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]
    return size


def y_to_one_hot(y):
    n = np.unique(y).size
    one_hot_y = np.zeros((y.size, n))
    for i, val in enumerate(np.unique(y)):
        matching_indices = np.asarray(np.where(y == val))
        one_hot_y[:, i][matching_indices] = 1
    return one_hot_y


def softmax_to_one_hot(y):
    try:
        y = y.detach().numpy()
    except:
        pass
    argmax = np.argmax(y, axis=1)
    one_hot_y = np.zeros(y.shape)
    one_hot_y[np.arange((one_hot_y.shape[0])), argmax] = 1
    return one_hot_y


def get_last_dim_of_conv_network(layers: int, dim: int, pooling: bool):
    for _ in range(layers):
        dim = dim - 2
        if pooling:
            dim = dim // 2
    return dim
