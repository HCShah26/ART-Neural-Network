from Neural_Network_Class import NeuralNetwork

if __name__ == "__main__":
    # Define the structure of the neural network (e.g., 2 input neurons, 3 hidden neurons, 1 output neuron)
    layers = [2, 3, 1]

    nn = NeuralNetwork(layers)

    # Input values
    inputs = [0.5, 0.9]

    # Compute the output of the neural network
    outputs = nn.compute(inputs)

    print("Output:", outputs)
