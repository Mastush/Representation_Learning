import representation, svm, evaluation
from data_loading import RepresentableVectorDataset
from networks import SimpleConvNetwork, SimpleNetwork

import numpy as np


def add_network_to_vector_representation_rep(dataset: RepresentableVectorDataset, input_shape, q: int, n_train: int,
                                             dim_reduction: int = None, network_type: str = "simple",
                                             return_output_dim: bool = True):
    networks_dict = {"simple": SimpleNetwork, "conv": SimpleConvNetwork}  # TODO: write get network function
    assert network_type.lower() in networks_dict, "Network type {} not supported".format(network_type)
    network = networks_dict[network_type](input_shape, q)
    nn_rep = representation.SimpleNetworkGradientRepresentation(network) if network_type.lower() == "simple" \
        else representation.SimpleConvNetworkGradientRepresentation(network)  # TODO: write get net rep function
    dataset.append_representation(nn_rep)
    x, y = dataset.get_training_examples(n_train, dim_reduction=dim_reduction)
    w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=0.0001,
                                 max_iter=1000)  # TODO: smart reg
    x, y = dataset.get_training_examples(n_train, dim_reduction=dim_reduction)

    # TODO: make this prettier, write a func for it
    x_test, y_test = dataset.get_test_examples(None, dim_reduction=dim_reduction)
    performance = evaluation.evaluate_model(w, x, y)
    print("train performance is {}".format(performance))
    performance = evaluation.evaluate_model(w, x_test, y_test)
    print("test performance is {} \n\n".format(performance))

    w_rep = representation.MatrixRepresentation(w.get_w()[:-1], x.shape[1:], 1)
    dataset.append_representation(w_rep)

    if return_output_dim:
        return q


def add_network_to_vector_rounds(n_rounds: int, dataset: RepresentableVectorDataset, input_shape, q: int,
                                 n_train: int = None, dim_reduction: int = None, network_type: str = "simple",
                                 return_output_dim: bool = True):
    output_dim = None
    for i in range(n_rounds):
         output_dim = add_network_to_vector_representation_rep(dataset, input_shape if i == 0 else output_dim, q - i,
                                                               n_train, dim_reduction, network_type)
        # q - i is because of q=d bug. TODO: fix q=d bug
    if return_output_dim:
        return output_dim
