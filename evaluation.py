import numpy as np
import torch.nn as nn

import utils
import svm


def zero_one_loss(pred, truth):
    return 1 - accuracy(pred, truth)


def accuracy(pred, truth):
    pred = np.asarray(pred).squeeze()
    truth = np.asarray(truth)
    return np.sum(pred == truth) / truth.size


def mse(pred, truth):
    return np.mean((pred - truth) ** 2)


def evaluate_model(model, x, y, eval_f=accuracy, pred_postprocessing=None, out_dim: int = 1, batch_size: int = None):
    x = np.asarray(x)
    y = np.asarray(y)
    if isinstance(model, nn.Module):
        model.eval()
        batch_size = 1 if batch_size is None else batch_size
        pred = None
        for i in range((x.shape[0] // batch_size) + 1):
            x_for_network = x[i * batch_size:(i + 1) * batch_size]
            if x_for_network.size > 0:
                batch_pred = utils.safe_tensor_to_ndarray(model(x_for_network))
                pred = batch_pred if pred is None else np.concatenate((pred, batch_pred))
    else:  # then isinstance(model, svm.SVMWrapper) == True
        pred = np.asarray([model(x[i], single_example=True) for i in range(y.shape[0])])

    if pred_postprocessing is not None:
        pred = pred_postprocessing(pred)

    return eval_f(pred, y)

