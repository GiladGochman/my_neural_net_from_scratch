def train_network(nn, dataset, alpha=0.3):
    """
    nn: the neural network object with weights and thresholds
    dataset: list of pairs (input_vector, target_vector) in float form 0.0 or 1.0
    alpha: learning rate
    """
    best_error_count = float('inf')
    no_improvement_streak = 0
    error_history = []  # for plotting

    epoch = 0
    while epoch < 10000:  # limit epochs to avoid infinite loop
        epoch += 1
        total_abs_error = 0
        current_epoch_errors = 0

        # Initialize accumulators for this epoch
        weight_updates = [[[0.0 for i in range(len(layer[j]))] for j in range(len(layer))] for layer in nn.weights]
        threshold_updates = [[0.0 for j in range(len(layer))] for layer in nn.thresholds]
        print(f"Epoch {epoch} Starting...")  # Debugging statement
        # print(f" Weights: {nn.weights}")
        # print(f" Thresholds: {nn.thresholds}")

        for inputs, targets in dataset:
            activations = nn.forward(inputs)
            output = activations[-1]

            row_error = sum(abs(t - o) for t, o in zip(targets, output))
            total_abs_error += row_error

            deltas = [[] for _ in range(nn.num_layers)]

            output_layer_idx = nn.num_layers - 1
            for i in range(nn.layer_sizes[output_layer_idx]):
                o = output[i]
                t = targets[i]
                delta = (t - o) * o * (1 - o)
                # print(f"Output Neuron {i}: Target = {t:.4f}, Output = {o:.4f}, Delta = {delta:.4f}")
                deltas[output_layer_idx].append(delta)

            for l in range(output_layer_idx - 1, 0, -1):
                for i in range(nn.layer_sizes[l]):
                    error_sum = 0
                    for j in range(nn.layer_sizes[l + 1]):
                        error_sum += deltas[l + 1][j] * nn.weights[l][j][i]
                    val = activations[l][i]
                    delta = error_sum * val * (1 - val)
                    deltas[l].append(delta)
                    # print(f"Hidden Layer {l} Neuron {i}: Activation = {val:.4f}, Delta = {delta:.4f}")

            for l in range(len(nn.weights)):
                for j in range(len(nn.weights[l])):
                    for i in range(len(nn.weights[l][j])):
                        weight_updates[l][j][i] += alpha * deltas[l + 1][j] * activations[l][i]
                    threshold_updates[l][j] -= alpha * deltas[l + 1][j]
                    # print(f"Layer {l} Neuron {j}: Weight Updates = {weight_updates[l][j]}, Threshold Update = {threshold_updates[l][j]:.4f}")

        current_epoch_errors = total_abs_error
        # print("=== FINAL ACCUMULATED UPDATES ===")
        # for l in range(len(threshold_updates)):
        #     for j in range(len(threshold_updates[l])):
                # print(f"Layer {l} Neuron {j}: Final Threshold Update = {threshold_updates[l][j]:.6f}")

        # print(f"Epoch {epoch}: Total Abs Error = {total_abs_error:.4f}, Error Count = {current_epoch_errors}")

        for l in range(len(nn.weights)):
            for j in range(len(nn.weights[l])):
                for i in range(len(nn.weights[l][j])):
                    nn.weights[l][j][i] += weight_updates[l][j][i]
                nn.thresholds[l][j] += threshold_updates[l][j]
                # print(f"After Epoch {epoch} Update - Layer {l} Neuron {j}: Weights = {nn.weights[l][j]}, Threshold = {nn.thresholds[l][j]:.4f}")

        error_history.append(current_epoch_errors)

        if total_abs_error < 0.1:
            save_model(nn, "model.txt")   # ← save on convergence
            return True, error_history

        if current_epoch_errors < best_error_count:
            best_error_count = current_epoch_errors
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

        if no_improvement_streak >= 5:
            return False, error_history

    return False, error_history


def save_model(nn, filepath="model.txt"):
    """
    Save converged weights and thresholds to a text file.

    File format:
        Line 1 : number of weight layers  (= num_layers - 1)
        Line 2 : all layer sizes separated by spaces
        Then, for every weight layer l:
            one row per neuron j  →  the weights connecting layer l to neuron j in layer l+1
        Then, for every weight layer l:
            one row for all threshold values of that layer, space-separated
    """
    num_weight_layers = len(nn.weights)           # num_layers - 1

    with open(filepath, "w") as f:
        # Line 1 – number of weight layers
        f.write(f"{num_weight_layers}\n")

        # Line 2 – layer sizes
        f.write(" ".join(str(s) for s in nn.layer_sizes) + "\n")

        # Weights: for each layer, one row per neuron
        for l in range(num_weight_layers):
            for j in range(len(nn.weights[l])):
                row = " ".join(f"{w:.4f}" for w in nn.weights[l][j])
                f.write(row + "\n")

        # Thresholds: for each layer, one row with all values
        for l in range(num_weight_layers):
            row = " ".join(f"{t:.4f}" for t in nn.thresholds[l])
            f.write(row + "\n")

    print(f"Model saved to '{filepath}'.")