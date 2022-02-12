import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
        # number of rows corresponds to number of inputs
        # number of columns corresponds to number of neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # since we only need one bias per neuron, we create a 1xn_neurons array
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        