import math
import sys

def sigmoid(x):
    if x < -100: return 0.0
    if x > 100: return 1.0
    return 1 / (1 + math.exp(-x))

def load_model(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    idx = 0
    num_weight_layers = int(lines[idx]); idx += 1
    layer_sizes = list(map(int, lines[idx].split())); idx += 1
    
    weights = []
    thresholds = []
    
    for l in range(num_weight_layers):
        rows = layer_sizes[l + 1]  # number of neurons in next layer
        weight_matrix = []
        for _ in range(rows):
            row = list(map(float, lines[idx].split())); idx += 1
            weight_matrix.append(row)
        weights.append(weight_matrix)
        
        thresh = list(map(float, lines[idx].split())); idx += 1
        thresholds.append(thresh)
    
    return weights, thresholds

def forward(inputs, weights, thresholds):
    current = inputs
    for layer_idx in range(len(weights)):
        next_layer = []
        for j in range(len(weights[layer_idx])):
            net = sum(weights[layer_idx][j][k] * current[k] for k in range(len(current)))
            z = net - thresholds[layer_idx][j]
            next_layer.append(sigmoid(z))
        current = next_layer
    return current

def parse(val):
    if val.upper() == 'T': return 1.0
    if val.upper() == 'F': return 0.0
    raise ValueError(f"Invalid input '{val}', use T or F")

if len(sys.argv) < 3:
    print("Usage: python script.py <input1> <input2> ...")
    print("Example: python script.py T F T")
    sys.exit(1)

weights, thresholds = load_model("model.txt")
inputs = [parse(v) for v in sys.argv[1:]]

output = forward(inputs, weights, thresholds)
predicted = ['T' if round(o) == 1 else 'F' for o in output]

print(f"Input:  {' '.join(sys.argv[1:])}")
print(f"Output: {' '.join(predicted)}")
