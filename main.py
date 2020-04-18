import representation
import data_loading
import networks
import svm
import evaluation

n = None
d = 50

q = d
model = networks.SimpleNetwork(d, q)
nn_rep = representation.SimpleNetworkGradientRepresentation(model)
mnist_dataset = data_loading.RepresentableMnist([nn_rep])
# mnist_dataset = data_loading.RepresentableMnist([])

x, y = mnist_dataset.get_training_examples(n, dim_reduction=d)
x_test, y_test = mnist_dataset.get_test_examples(n, dim_reduction=d)
print("Collected datasets")
print("data shape is: {}".format(x.shape))
w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=0.000001, max_iter=1000)
print("Acquired linear separator")
performance = evaluation.evaluate_model(w, x, y)
print("train preformance is {}".format(performance))
performance = evaluation.evaluate_model(w, x_test, y_test)
print("test preformance is {}".format(performance))


print("Getting next representation and updating datasets...")
w_rep = representation.MatrixRepresentation(w.get_w()[-1], x.shape[1:], 1)
mnist_dataset.append_representation(w_rep)
x, y = mnist_dataset.get_training_examples(n, dim_reduction=d)
x_test, y_test = mnist_dataset.get_test_examples(n, dim_reduction=d)
print("Collected datasets")
print("data shape is: {}".format(x.shape))
w = svm.get_linear_separator(x, y, type_of_classifier='sdca', verbose=2, alpha=0.0000000001, max_iter=10000)
print("Acquired linear separator")
performance = evaluation.evaluate_model(w, x, y)
print("train preformance is {}".format(performance))
performance = evaluation.evaluate_model(w, x_test, y_test)
print("test preformance is {}".format(performance))
