import numpy as np


def zero_one_loss(pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    return np.sum(pred == truth) / truth.size


def evaluate_model(model, x, y, loss=zero_one_loss):
    x = np.asarray(x)
    y = np.asarray(y)
    try:
        pred = model(x)
    except:
        pred = [model(x[i]) for i in range(y.size)]
    return loss(pred, y)

