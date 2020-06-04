import rounds, data_loading, evaluation, svm, networks


import numpy as np

n = 10000
d = 50
dim_red = d

q = 50


mnist_dataset = data_loading.RepresentableMnist([], normalize_raw=True, normalize_reps=True)
output_dim = rounds.add_network_to_vector_rounds(2, mnist_dataset, d, q, n, dim_red, network_type="simple")
# output_dim = rounds.add_network_to_vector_rounds(2, mnist_dataset, (1, 1, 28, 28), q, n, network_type="conv")
# TODO: notice that we would get d=q bug here in the 2nd round!!!!!!! fix!!!!!!

print("Finished getting Representations")

print("Getting represented dataset...")

x, y = mnist_dataset.get_training_examples(n, dim_reduction=dim_red)
x_test, y_test = mnist_dataset.get_test_examples(n, dim_reduction=dim_red)

print("Going into final linear separator")

w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=0.0001, max_iter=1000)

performance = evaluation.evaluate_model(w, x, y)
print("train performance is {}".format(performance))
performance = evaluation.evaluate_model(w, x_test, y_test)
print("test performance is {}".format(performance))


# mnist_dataset = data_loading.RepresentableMnist([])
# x, y = mnist_dataset.get_training_examples(n, dim_reduction=dim_red)
# x_test, y_test = mnist_dataset.get_test_examples(n, dim_reduction=dim_red)
# model = networks.ConvNetwork((1, 1, 28, 28), 32, 2)
# networks.train_network(model, x, y, x_test, y_test, epochs=300, y_postprocessing=None)
