import torch
import numpy as np


EPSILON = 0.0000001


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_tensor_to_ndarray(tensor: torch.Tensor):
    return np.copy(tensor.data.cpu().numpy())


def unison_shuffled_copies(a: np.ndarray, b: np.ndarray):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def normalize_vectors(x: np.ndarray):
    flat_x = flatten_data(x)  # TODO: fix for 3 dims
    mags = np.linalg.norm(flat_x, axis=1) + EPSILON
    return np.divide(x.T, mags).T


def flatten_data(data, is_tensor=False):
    last_dim = 1
    for i in range(1, len(data.shape)):
        last_dim *= data.shape[i]
    if is_tensor:
        return data.view(data.shape[0], last_dim)
    return np.reshape(data, (data.shape[0], last_dim))


def standardize_data(x):
    mean, std = x.mean(axis=0), x.std(0)
    x -= mean
    x /= std + EPSILON
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


def get_last_dim_of_conv_network(layers: int, dim: int, pooling: bool, kernel_size: int, padded: bool):
    for _ in range(layers):
        if not padded:
            dim = dim - kernel_size // 2
        if pooling:
            dim = dim // 2
    return dim


def pad_image(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                          mode='constant', constant_values=0))


def image_to_patches(im, patch_size):
    c, h, w = im.shape
    patch_r = patch_size // 2
    # this saves spatial relations
    patches = np.zeros((h - 2 * patch_r, w - 2 * patch_r, c, patch_size, patch_size))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patches[i, j, ...] = im[:, i:i + patch_size, j:j + patch_size]
    return patches


def get_activation_gradient(activation, x, return_ndarray=False):
    x = torch.from_numpy(x).to(get_device())
    x.requires_grad = True
    y = activation(x).sum()
    y.backward()
    return safe_tensor_to_ndarray(x.grad) if return_ndarray else x.grad


def get_multiclass_to_binary_truth_f(n_classes):

    def multiclass_to_binary_truth(truth):
        truth = (truth.astype(np.int) > n_classes // 2).astype(np.int)
        truth[truth == 0] = -1
        return truth

    return multiclass_to_binary_truth


def print_args(args):
    for key, val in args.__dict__.items():
        print("{} = {}".format(key, val))