from torch.nn import BCELoss, MSELoss
from torch.optim import SGD, Adam, Adagrad
import numpy as np
from torch import from_numpy

import argparse

import evaluation, data_loading, utils
from networks import FCNetwork, ConvNetwork


def train_network(model, x, y, x_test=None, y_test=None, epochs=50, batch_size=64, loss_f=BCELoss,
                  optimizer=Adam, lr=0.001, y_postprocessing=utils.y_to_one_hot, weight_decay: float = 0.0000001,
                  verbose: int = 1):
    print("Started training")

    best_epoch = (0, 0, 0)

    if y_postprocessing is not None:
        y = y_postprocessing(y)
        y_test = y_postprocessing(y_test)
    y, y_test = y.astype(np.float), y_test.astype(np.float)

    loss_f = loss_f()
    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        for i in range((x.shape[0] // batch_size) + 1):
            x_for_network = x[i * batch_size:(i + 1) * batch_size]
            y_for_network = y[i * batch_size:(i + 1) * batch_size]
            if x_for_network.size == 0:
                break
            if verbose > 0:
                print("batch {} of {}".format(i + 1, x.shape[0] // batch_size + 1))
            optimizer.zero_grad()
            pred = model(x_for_network)
            loss = loss_f(pred, from_numpy(y_for_network).float())
            loss.backward()
            optimizer.step()
        train_performance = evaluation.evaluate_model(model, x, y, pred_postprocessing=utils.softmax_to_one_hot,
                                                      out_dim=2, batch_size=batch_size)
        print("train performance is {}".format(train_performance))
        if x_test is not None:
            performance = evaluation.evaluate_model(model, x_test, y_test, pred_postprocessing=utils.softmax_to_one_hot,
                                                    out_dim=2, batch_size=batch_size)
            print("test performance is {}".format(performance))
            if performance > best_epoch[-1]:
                best_epoch = (epoch, train_performance, performance)
        print("Finished epoch {}".format(epoch))
    if x_test is not None:
        print("Finished training. Best epoch is {} with training performance {} "
              "and test performance {}".format(*best_epoch))


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
    parser.add_argument('-k', '--kernel_size', type=int, default=3, help="Kernel size for convolutional networks")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Number of examples per batch")
    parser.add_argument('--normalize_raw', action='store_true',
                        default=False, help="Whether or not to normalize the raw datapoints.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate for network training")
    parser.add_argument('-o', '--optimizer', type=str, default="adam",
                        help="The type of optimizer for network training")
    parser.add_argument('-w', '--weight_decay', type=float, default=0.0000001,
                        help="Coefficient for weight decay in training")
    parser.add_argument('-v', '--verbose', type=int, default=0, help="Should be 1 to see batch progression.")
    args = parser.parse_args()

    if args.network_type == 'conv' and args.dim_red is not None:
        print("Dimensionality reduction is not used for convolutional networks!")
        args.dim_red = None

    args.optimizer = get_optimizer(args.optimizer)

    return args


def get_optimizer(opt_str):
    optimizer_dict = {"adam": Adam, "adagrad": Adagrad, "sgd": SGD}
    assert opt_str.lower() in optimizer_dict, "Optimizer of type {} not supported".format(opt_str)
    return optimizer_dict[opt_str.lower()]


def main_nn():
    args = get_network_training_args()
    dataset = data_loading.get_dataset(args.dataset, args.normalize_raw, False)
    if args.network_type == 'simple':
        d = args.dim_red if args.dim_red is not None else dataset.get_raw_input_shape()
        model = FCNetwork(d, args.neurons, args.layers)
    else:  # meaning args.network_type == 'conv'
        input_shape = dataset.get_raw_input_shape(True)
        model = ConvNetwork(input_shape, args.neurons, args.layers, args.kernel_size, auto_pad=True)

    model.to(utils.get_device())
    print("Model device is {}".format(utils.get_device()))

    x, y = dataset.get_training_examples(args.n_train, False, args.dim_red if args.network_type == 'simple' else None)
    x_test, y_test = dataset.get_test_examples(args.n_test, False,
                                               args.dim_red if args.network_type == 'simple' else None)
    train_network(model, x, y, x_test, y_test, args.epochs, args.batch_size, optimizer=args.optimizer,
                  lr=args.learning_rate, weight_decay=args.weight_decay, verbose=args.verbose)


if __name__ == '__main__':
    main_nn()
