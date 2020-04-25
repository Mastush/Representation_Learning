import rounds, data_loading, evaluation, svm


import numpy as np

n = 10000
d = 28
dim_red = d

q = 25

mnist_dataset = data_loading.RepresentableMnist([], standardize_all=False)
output_dim = rounds.add_network_to_vector_rounds(1, mnist_dataset, d, q, n, dim_red, network_type="simple")
# TODO: notice that we would get d=q bug here in the 2nd round!!!!!!! fix!!!!!!
x, y = mnist_dataset.get_training_examples(n, dim_reduction=dim_red)
x_test, y_test = mnist_dataset.get_test_examples(n, dim_reduction=dim_red)

w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=0.0001, max_iter=1000)

performance = evaluation.evaluate_model(w, x, y)
print("train performance is {}".format(performance))
performance = evaluation.evaluate_model(w, x_test, y_test)
print("test performance is {}".format(performance))



#
# # model = networks.SimpleConvNetwork(3, 1000, (1, 28, 28))
# model = networks.SimpleNetwork(d, q)
# # nn_rep = representation.SimpleConvNetworkGradientRepresentation(model)
# nn_rep = representation.SimpleNetworkGradientRepresentation(model)
# mnist_dataset = data_loading.RepresentableMnist([])
# # mnist_dataset = data_loading.RepresentableMnist([])
#
# x, y = mnist_dataset.get_training_examples(n, dim_reduction=d)
# x_test, y_test = mnist_dataset.get_test_examples(n, dim_reduction=d)
# print("-------------- {}---------------".format(np.linalg.norm(x[0])))
# print("Collected datasets")
# print("data shape is: {}".format(x.shape))
# w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=0.0001, max_iter=1000)
# print("Acquired linear separator")
# performance = evaluation.evaluate_model(w, x, y)
# print("train performance is {}".format(performance))
# performance = evaluation.evaluate_model(w, x_test, y_test)
# print("test performance is {}".format(performance))
#
#
# print("Getting next representation and updating datasets...")
# w_rep = representation.MatrixRepresentation(w.get_w(), x.shape[1:], 1)
# # w_rep = representation.MatrixRepresentation(w.get_w()[:-1], x.shape[1:], 1)
# mnist_dataset.append_representation(w_rep)
# x, y = mnist_dataset.get_training_examples(n, dim_reduction=d)
# print("-------------- {}---------------".format(np.linalg.norm(x[0])))
# x_test, y_test = mnist_dataset.get_test_examples(n, dim_reduction=d)
# print("Collected datasets")
# print("data shape is: {}".format(x.shape))
# w2 = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=0.00000001, max_iter=1)
# print("Acquired linear separator")
# performance = evaluation.evaluate_model(w2, x, y)
# print("train performance is {}".format(performance))
# performance = evaluation.evaluate_model(w2, x_test, y_test)
# print("test performance is {}".format(performance))
# a = 0.000001
# b = 5297
#
# w2._model.coef_ = np.ones(w2._model.coef_.shape)
# # w2._model.coef_[0, -1] = w._model.coef_[0, -1]
# performance = evaluation.evaluate_model(w2, x, y)
# print("train performance is {}".format(performance))
# performance = evaluation.evaluate_model(w2, x_test, y_test)
# print("test performance is {}".format(performance))
#

