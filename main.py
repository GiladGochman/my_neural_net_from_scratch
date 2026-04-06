import NeuralNetwork
import train
import os
import matplotlib.pyplot as plt

def parse_config(filename):
    """parsing the dimentions of the neural net from the file network_config.txt"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        num_layers = int(lines[0])
        layer_sizes = [int(x) for x in lines[1:num_layers+1]]
    return layer_sizes

def parse_task(filename, num_inputs, num_outputs):
    """parsing the truth table from file task_data.txt and converting it to numbers (0 and 1)"""
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()   #splitting each line to space delimiter
            if not parts: continue
            values = [1.0 if val == 'T' else 0.0 for val in parts]
            # splitting to inputs and outputs according to the dimentions given network_config.txt
            inputs = values[:num_inputs]
            targets = values[num_inputs:]
            dataset.append((inputs, targets))
    return dataset

def run():
    
    # load inputs
    if not os.path.exists('network_config.txt') or not os.path.exists('task_data.txt'):
        print("Error: Input files missing!")
        return
    layer_sizes = parse_config('network_config.txt')
    num_in = layer_sizes[0]
    num_out = layer_sizes[-1] #output dimention is in the last cell    
    training_data = parse_task('task_data.txt', num_in, num_out)
    
    print(f"Starting Training for structure: {layer_sizes}")
    print(f"Alpha: 0.3, Range: [0,1]")
    print("-" * 30)

    # loop 10 times to try different random initializations of weights and thresholds
    final_success = False
    for attempt in range(1, 11): # change to 11 for final version
        print(f"Attempt #{attempt}: Initializing random weights/thresholds...")
        
        # initialize the neural network with random weights and thresholds
        nn = NeuralNetwork.NeuralNetwork(layer_sizes)
        
        isSuccess, error_history = train.train_network(nn, training_data, alpha=0.3)
        
        if isSuccess:
            print(f"SUCCESS! Network converged in attempt {attempt}.")
            print(f"Final error count: 0")
            final_success = True
            
            print("\nFinal Truth Table Results:")
            # using all the inputs to show the final predictions after training:
            for inputs, targets in training_data:
                prediction = nn.get_output(inputs)
                # rounding results to 3 places after decimal point:
                clean_pred = [round(p, 3) for p in prediction]
                print(f"In: {inputs} | Target: {targets} | Predicted: {clean_pred}")
            
            # Plot error history for the report
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(error_history) + 1), error_history, marker='o', linestyle='-')
            plt.title(f'Error vs. Training Sessions (Attempt {attempt})')
            plt.xlabel('Training Sessions (Epochs)')
            plt.ylabel('Total Absolute Error')
            plt.grid(True)
            plt.savefig(f'error_plot_attempt_{attempt}.png')
            plt.show()
            
            break
        else:
            print(f"Attempt {attempt} failed (Stuck in local optimum). Restarting...")
    
    if not final_success:
        print("\nFailed to converge after 10 random restarts.")

if __name__ == "__main__":
    run()