import numpy as np


class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias with random values
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def activate(self, inputs):
        # Compute the weighted sum of inputs
        z = np.dot(inputs, self.weights) + self.bias

        # Apply the activation function (e.g., sigmoid)
        return self.sigmoid(z)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
