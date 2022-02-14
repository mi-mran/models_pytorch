import numpy as np
from sklearn.covariance import log_likelihood

class Dense:
    def __init__(self, n_inputs, n_neurons):
        # number of rows corresponds to number of inputs
        # number of columns corresponds to number of neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # since we only need one bias per neuron, we create a 1xn_neurons array
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        # axis=1 ensures max function applied row-wise
        # keepdims=True ensures max function outputs a column array
        # minus max value to prevent exploding value neurons
        exponential_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # normalised probabilities
        probability_vals = exponential_vals / np.sum(exponential_vals, axis=1, keepdims=True)

        self.output = probability_vals

class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)

        loss = np.mean(sample_losses)

        return loss

class LossCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e7)

        # logic for checking if y_true is stored as one-hot encoded or categorical labels

        if len(y_true.shape) == 2:
            # one-hot encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        elif len(y_true.shape) == 1:
            # categorical labels
            # range(n_samples) gets all rows in the number of samples in the batch
            # y_true contains the column index for retrieving the confidence values for all samples
            correct_confidences = y_pred_clipped[range(n_samples), y_true]

        log_likelihood = -np.log(correct_confidences)

        return log_likelihood