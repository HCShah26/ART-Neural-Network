from Layer_Class import Layer


class NeuralNetwork:
    def __init__(self, layers):
        # Create a list of layers
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i + 1], layers[i]))

    def compute(self, inputs):
        outputs = inputs

        # Pass inputs through each layer
        for layer in self.layers:
            outputs = layer.process_inputs(outputs)

        return outputs
