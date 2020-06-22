from torch.nn import BCELoss, MSELoss
from torch.optim import SGD, Adam, Adagrad
import numpy as np
from torch import from_numpy

import argparse

import evaluation, data_loading, utils
from networks import FCNetwork, ConvNetwork


def train_network(model, x, y, x_test=None, y_test=None, epochs=50, batch_size=64, loss_f=BCELoss,
                  optimizer=Adam, lr=0.001, y_postprocessing=utils.y_to_one_hot, weight_decay: float = 0.0000001):
    print("Started training")

    if y_postprocessing is not None:
        y = y_postprocessing(y)
        y_test = y_postprocessing(y_test)
    y, y_test = y.astype(np.float), y_test.astype(np.float)

    loss_f = loss_f()
    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        for i in range(x.shape[0] // batch_size):
            print("batch {} of {}".format(i + 1, x.shape[0] // batch_size))
            optimizer.zero_grad()
            x_for_network = x[i * batch_size:(i + 1) * batch_size]
            y_for_network = y[i * batch_size:(i + 1) * batch_size]
            pred = model(x_for_network)
            loss = loss_f(pred, from_numpy(y_for_network).float())
            loss.backward()
            optimizer.step()
        performance = evaluation.evaluate_model(model, x, y, pred_postprocessing=utils.softmax_to_one_hot, out_dim=2)
        print("train performance is {}".format(performance))
        print("Finished epoch {}".format(epoch))
        if x_test is not None:
            performance = evaluation.evaluate_model(model, x_test, y_test, pred_postprocessing=utils.softmax_to_one_hot, out_dim=2)
            print("test performance is {}".format(performance))


def get_network_training_args():
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'], help='Which dataset to use')
    parser.add_argument('-d', '--dim_red', type=int, default=None, help='Dimensionality reduction target dimension')
    parser.add_argument('-q', '--neurons', type=int, help='The number of neurons to use in each hidden layers')
    parser.add_argument('-l', '--layers', type=int, default=1, help="The number of hidden layers")
    parser.add_argument('--n_train', type=int, default=None, help="The number of examples to use for training. "
                                                                  "None means the entire dataset.")
    parser.add_argument('--n_test', type=int, default=None, help="The number of examples to use for testing. "
                                                                 "None means the entire dataset.")
    parser.add_argument('--network_type', type=str, choices=['simple', 'conv'])
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="Number of examples per batch")
    parser.add_argument('--normalize_raw', action='store_true',
                        default=False, help="Whether or not to normalize the raw datapoints.")
    args = parser.parse_args()

    if args.network_type == 'conv' and args.dim_red is not None:
        print("Dimensionality reduction is not used for convolutional networks!")
        args.dim_red = None
    return args


def main_nn():
    args = get_network_training_args()
    dataset = data_loading.get_dataset(args.dataset, args.normalize_raw, False)
    if args.network_type == 'simple':
        d = args.dim_red if args.dim_red is not None else dataset.get_raw_input_shape()
        model = FCNetwork(d, args.neurons, args.layers)
    else:  # meaning args.network_type == 'conv'
        input_shape = dataset.get_raw_input_shape(True)
        model = ConvNetwork(input_shape, args.neurons, args.layers)
    x, y = dataset.get_training_examples(args.n_train, False, args.dim_red if args.network_type == 'simple' else None)
    x_test, y_test = dataset.get_test_examples(args.n_test, False,
                                               args.dim_red if args.network_type == 'simple' else None)
    train_network(model, x, y, x_test, y_test, args.epochs, args.batch_size)


if __name__ == '__main__':
    main_nn()
