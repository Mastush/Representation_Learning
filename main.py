import representation
import data_loading
import networks
import svm
import evaluation

# d = 50
# q = d
# model = networks.SimpleNetwork(d, q)
# nn_rep = representation.SimpleNetworkGradientRepresentation(model)
# mnist_dataset = data_loading.RepresentableMnist([nn_rep])
d = None
mnist_dataset = data_loading.RepresentableMnist([])
x, y = mnist_dataset.get_training_examples(None, dim_reduction=d)
print(x.shape)
x_test, y_test = mnist_dataset.get_test_examples(None, dim_reduction=d)
print("Collected datasets")
w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=1, max_iter=1000)
print("Acquired linear separator")
performance = evaluation.evaluate_model(w, x, y)
print(performance)
performance = evaluation.evaluate_model(w, x_test, y_test)
print(performance)
