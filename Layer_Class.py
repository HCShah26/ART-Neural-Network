from Neuron_Class import Neuron


class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        # Create a list of neurons
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def process_inputs(self, inputs):
        # Collect outputs from all neurons in the layer
        outputs = [neuron.activate(inputs) for neuron in self.neurons]
        return outputs
