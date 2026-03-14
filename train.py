
def train_network(nn, dataset, alpha=0.3):
    """
    nn: the neural network object with weights and thresholds
    dataset: list of pairs (input_vector, target_vector) in float form 0.0 or 1.0
    alpha: learning rate
    """
    best_error_count = len(dataset) + 1
    no_improvement_streak = 0
    error_history = [] # for plotting
    
    epoch = 0
    while epoch < 1000000000: #limit epochs to avoid infinite loop
        epoch += 1
        total_abs_error = 0
        current_epoch_errors = 0
        
        for inputs, targets in dataset:
            activations = nn.forward(inputs) # calculate all values in neural network
            output = activations[-1]
            
            # calculate error of outputs for single entry of inputs and targets
            row_error = sum(abs(t - o) for t, o in zip(targets, output))
            total_abs_error += row_error
            
            if row_error >= 0.1:
                current_epoch_errors += 1
            deltas = [[] for _ in range(nn.num_layers)] #initialize list of lists for deltas for each node 
            #calculate deltas for output layer:
            output_layer_idx = nn.num_layers - 1
            for i in range(nn.layer_sizes[output_layer_idx]):
                o = output[i]
                t = targets[i]
                # formula: delta = (Target - Output) * Output * (1 - Output)
                delta = (t - o) * o * (1 - o)
                print(f"Output Neuron {i}: Target = {t:.4f}, Output = {o:.4f}, Delta = {delta:.4f}")

                deltas[output_layer_idx].append(delta)
            
            # calculate deltas for hidden layers using the caluculated deltas of the layer after them:
            for l in range(output_layer_idx - 1, 0, -1): #start in one before last layer, decrese one until the first hidden layer
                for i in range(nn.layer_sizes[l]):
                    error_sum = 0
                    for j in range(nn.layer_sizes[l+1]):
                        error_sum += deltas[l+1][j] * nn.weights[l][j][i]
                    
                    val = activations[l][i]
                    delta = error_sum * val * (1 - val)
                    deltas[l].append(delta)
            
            # update weights and thresholds using the calculated deltas:
            for l in range(len(nn.weights)):
                for j in range(len(nn.weights[l])):
                    for i in range(len(nn.weights[l][j])): 
                        #folmula: Wnew = Wold + alpha * delta * Input
                        nn.weights[l][j][i] += alpha * deltas[l+1][j] * activations[l][i]
                    
                    # formula: Tnew = Told - alpha * delta * const (constant is 1)
                    nn.thresholds[l][j] -= alpha * deltas[l+1][j]

        error_history.append(current_epoch_errors)
    
        if total_abs_error < 0.1:
            return True, error_history
        
        if current_epoch_errors < best_error_count:
            best_error_count = current_epoch_errors
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
            
        if no_improvement_streak >= 5:
            return False, error_history

        
    return False, error_history