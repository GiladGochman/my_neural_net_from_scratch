import math
import random

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: a list of dimentions for each layer
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # initialiaze random weights in range [0,1]
        self.weights = [] #vector of matrices, each matrix is the weights between layer i and i+1
        for i in range(self.num_layers - 1):
            cols = self.layer_sizes[i]   # cols = row length = size of current layer
            rows = self.layer_sizes[i+1] # rows = column length = size of next layer
            #create weights matrix for layer i to i+1 with random values in range [0,1]
            weight_matrix_k = [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
            self.weights.append(weight_matrix_k)
            
        # intialiaze random thresholds in range [0,1]
        self.thresholds = []
        for i in range(1, self.num_layers): # start from 1 because the input layer doesn't have thresholds
            num_neurons = self.layer_sizes[i]
            thresholds_vector_i = [random.uniform(-1, 1) for _ in range(num_neurons)]

            self.thresholds.append(thresholds_vector_i)

    def sigmoid(self, x):
        # print(f"Calculating sigmoid for x = {x:.4f}")
        if x < -100: return 0.0
        if x > 100: return 1.0
        sig = 1 / (1 + math.exp(-x))
        # print(f"Sigmoid output: {sig:.4f} for x = {x:.4f}")
        return sig

# forward pass function that calculates the entire activation list of vectors
    def forward(self, input_vector):

        activations = [input_vector] # will contain a list of all node value vectors after activation(for backpropagation), starting with the input layer
        current_layer_values = input_vector # the vector of values after activation for current layer
        # for each neuron j in layer i
        for i in range(len(self.weights)):
            next_layer_values = []            
            for j in range(len(self.weights[i])): 
                # matrix multiplication:
                net_input = sum(self.weights[i][j][k] * current_layer_values[k] 
                                for k in range(len(current_layer_values)))
                
                # subtract the threshold for this neuron:
                z = net_input - self.thresholds[i][j]
                # print(f"Layer {i+1} Neuron {j}: Net input = {net_input:.4f}, Threshold = {self.thresholds[i][j]:.4f}, Z = {z:.4f}"  )
                # use activevation function
                next_layer_values.append(self.sigmoid(z))
            
            current_layer_values = next_layer_values
            activations.append(current_layer_values)
        #print inputs and outputs for debugging:
        # print(f"input:", input_vector)
        # print(f"output:", current_layer_values)
        return activations

# the get_output function is called to show only the output (last vector of activations) for user friendlyness
    def get_output(self, input_vector):
        return self.forward(input_vector)[-1]