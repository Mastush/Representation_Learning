import torch
import numpy as np


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def safe_tensor_to_ndarray(tensor):
    return np.copy(tensor.data.cpu().numpy())
