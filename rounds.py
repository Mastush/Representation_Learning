import representation, svm
from data_loading import RepresentableVectorDataset
import networks, utils
import time


def is_convolutional(nn):
    conv_nns = [networks.SimpleConvNetwork]
    return type(nn) in conv_nns


def add_network_to_vector_representation_rep(dataset: RepresentableVectorDataset, input_shape, q: int, n_train: int,
                                             max_iter: int, alpha: float,
                                             dim_reduction: int = None, network_type: str = "simple",
                                             return_output_dim: bool = True):
    network_class = networks.get_network_for_reps(network_type)
    network = network_class(input_shape, q)

    print("Round network device is cuda? {}".format(next(network.parameters()).is_cuda))

    nn_reps = representation.get_net_rep(network, input_shape=input_shape)
    dataset.append_representation(nn_reps[0])
    x, y = dataset.get_training_examples(n_train, dim_reduction=dim_reduction, print_progress=True)
    # TODO: allow x, y to be batch generators

    w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=alpha, max_iter=max_iter)

    # this block is to allow the patches representation
    if len(nn_reps) > 1:
        dataset.remove_representations(1)
        dataset.append_representation(nn_reps[1])

    x, y = dataset.get_training_examples(1, dim_reduction=dim_reduction)

    # TODO: make this prettier, write a func for it
    # x_test, y_test = dataset.get_test_examples(None, dim_reduction=dim_reduction)
    # performance = evaluation.evaluate_model(w, x, y)
    # print("train performance is {}".format(performance))
    # performance = evaluation.evaluate_model(w, x_test, y_test)
    # print("test performance is {} \n\n".format(performance))

    w_rep = representation.get_separator_rep(dataset.get_last_representation(), w.get_w()[:-1], x[0].shape)
    # dataset.append_representation(w_rep)
    round_rep = representation.SequentialRepresentation([nn_reps[-1], w_rep])

    dataset.remove_representations(1)
    dataset.append_representation(round_rep)

    if return_output_dim:
        return dataset.get_last_representation().get_output_shape(x[0])


def add_network_to_vector_rounds(n_rounds: int, dataset: RepresentableVectorDataset, input_shape, q: list,
                                 max_iter_list: list, alpha_list: list,
                                 n_train: int = None, dim_reduction: int = None, network_type: str = "simple",
                                 return_output_dim: bool = True):
    output_dim = None
    for i in range(n_rounds):
        print("Starting round {}:".format(i + 1))
        start_time = time.time()
        output_dim = add_network_to_vector_representation_rep(dataset, input_shape if i == 0 else output_dim, q[i],
                                                              n_train, max_iter_list[i], alpha_list[i],
                                                              dim_reduction, network_type,
                                                              )
        print("Round {}/{} finished in {} minutes.\n".format(i + 1, n_rounds, (time.time() - start_time) // 60))
        # q - i is because of q=d bug. TODO: fix q=d bug
    if return_output_dim:
        return output_dim
