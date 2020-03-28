import representation
import data_loading
import networks

d = 50
q = 60
model = networks.SimpleNetwork(d, q)
nn_rep = representation.SimpleNetworkGradientRepresentation(model)
mnist = data_loading.RepresentableMnist([nn_rep])

x, y = mnist.get_training_examples(55, dim_reduction=d)