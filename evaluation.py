import numpy as np


def zero_one_loss(pred, truth):
    return 1 - accuracy(pred, truth)


def accuracy(pred, truth):
    pred = np.asarray(pred).squeeze()
    truth = np.asarray(truth)
    return np.sum(pred == truth) / truth.size


def mse(pred, truth):
    return np.mean((pred - truth) ** 2)


def evaluate_model(model, x, y, eval_f=accuracy, pred_postprocessing=None, out_dim: int = 1):
    x = np.asarray(x)
    y = np.asarray(y)
    pred = np.asarray([model(x[i], single_example=True) for i in range(y.shape[0])])  # TODO: write separate func for getting pred
    try:
        new_pred = np.zeros((pred.size, out_dim))
        for i in range(pred.shape[0]):
            new_pred[i] = pred[i].detach().numpy()
        pred = new_pred
    except:
        pass
    if pred_postprocessing is not None:
        pred = pred_postprocessing(pred)
    return eval_f(pred, y)

