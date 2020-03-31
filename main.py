import representation
import data_loading
import networks
import svm
import evaluation

d = 50
q = 60
model = networks.SimpleNetwork(d, q)
nn_rep = representation.SimpleNetworkGradientRepresentation(model)
mnist_dataset = data_loading.RepresentableMnist([nn_rep])

x, y = mnist_dataset.get_training_examples(200, dim_reduction=d)
w = svm.get_linear_separator(x, y)
performace = evaluation.evaluate_model(w, x, y)
a = 5