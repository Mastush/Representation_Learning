import rounds, data_loading, evaluation, svm, utils

import argparse

from resource import getrusage, RUSAGE_SELF
import time


def get_arguments():
    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('-r', '--rounds', type=int, help="The number of wanted representation rounds")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'], help='Which dataset to use')
    parser.add_argument('-d', '--dim_red', type=int, default=None, help='Dimensionality reduction target dimension')
    parser.add_argument('-q', '--neurons', type=int, nargs='+',
                        help='The number of neurons to use in the network\'s tangent kernel representation')
    parser.add_argument('--n_train', type=int, default=None, help="The number of examples to use for training. "
                                                                  "None means the entire dataset.")
    parser.add_argument('--n_test', type=int, default=None, help="The number of examples to use for testing. "
                                                                 "None means the entire dataset.")
    parser.add_argument('--network_type', type=str, choices=['simple', 'conv'])
    parser.add_argument('--normalize_raw', action='store_true',
                        default=False, help="Whether or not to normalize the raw datapoints.")
    parser.add_argument('--normalize_reps', default=False, action='store_true',
                        help="Whether or not to normalize after each representation.")
    parser.add_argument('-m', '--max_iter_optimization', nargs='+', type=int, default=[3000],
                        help="The maximum number of iterations for optimization per round.")
    parser.add_argument('-me', '--max_iter_evaluation', type=int, default=3000,
                        help="The maximum number of iterations for evaluation at the end.")
    parser.add_argument('-a', '--alpha_optimization', nargs='+', type=float, default=[0.000000001],
                        help="Regularization coefficient for optimization per round.")
    parser.add_argument('-ae', '--alpha_evaluation', type=float, default=0.000000001,
                        help="Regularization coefficient for evaluation at the end.")
    parser.add_argument('-c', '--cifar_path', type=str, default='/cs/dataset/CIFAR/cifar-10-batches-py/',
                        help="Path for CIFAR dataset.")
    args = parser.parse_args()

    if args.network_type == 'conv' and args.dim_red is not None:
        print("Dimensionality reduction is not used for convolutional networks!")
        args.dim_red = None

    if len(args.max_iter_optimization) == 1:
        args.max_iter_optimization = [*args.max_iter_optimization] * args.rounds
    elif len(args.max_iter_optimization) != args.rounds:
        raise ValueError("max_iter_optimization should have either 1 value or the same as the number of rounds!")

    if len(args.alpha_optimization) == 1:
        args.alpha_optimization = [*args.alpha_optimization] * args.rounds
    elif len(args.alpha_optimization) != args.rounds:
        raise ValueError("alpha_optimization should have either 1 value or the same as the number of rounds!")

    if len(args.neurons) == 1:
        q_list = []
        for i in range(args.rounds):
            q_list.append(args.neurons[0] - i)
        args.neurons = q_list
    elif len(args.neurons) != args.rounds:
        raise ValueError("neurons should have either 1 value or the same as the number of rounds!")

    return args


def main():
    args = get_arguments()
    utils.print_args(args)
    dataset = data_loading.get_dataset(args.dataset, args.normalize_raw, args.normalize_reps, args.cifar_path)

    print("Collected arguments and raw dataset.")

    if args.network_type == 'simple':
        input_shape = args.dim_red if args.dim_red is not None else dataset.get_raw_input_shape()
        rounds.add_network_to_vector_rounds(args.rounds, dataset, input_shape, args.neurons,
                                            args.max_iter_optimization, args.alpha_optimization, args.n_train,
                                            args.dim_red, network_type='simple')
    elif args.network_type == 'conv':
        input_shape = dataset.get_raw_input_shape(True)
        rounds.add_network_to_vector_rounds(args.rounds, dataset, input_shape, args.neurons,
                                            args.max_iter_optimization, args.alpha_optimization, args.n_train,
                                            None, network_type='conv')
    else:
        raise ValueError("Network type {} not supported".format(args.network_type))

    print("Finished getting Representations")
    print("Getting represented dataset:")
    print("Getting training examples...")
    x, y = dataset.get_training_examples(args.n_train, dim_reduction=args.dim_red, print_progress=True)
    print("Getting test examples...")
    x_test, y_test = dataset.get_test_examples(args.n_test, dim_reduction=args.dim_red, print_progress=True)

    print("Getting final linear separator")

    w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=args.alpha_evaluation,
                                 max_iter=args.max_iter_evaluation)

    performance = evaluation.evaluate_model(w, x, y)
    print("train performance is {}".format(performance))
    performance = evaluation.evaluate_model(w, x_test, y_test)
    print("test performance is {}".format(performance))


if __name__ == '__main__':
    print(getrusage(RUSAGE_SELF).ru_maxrss)
    start_time = time.time()
    main()
    print("Total memory used: {}".format(getrusage(RUSAGE_SELF).ru_maxrss))
    print("Total runtime: {} minutes".format((time.time() - start_time) // 60))
