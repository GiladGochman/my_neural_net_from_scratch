<!-- run instructions: -->

## how to run:

1. insert neural network dimetions to network_config.txt as such:

first row is the number of layers

rest of rows are the number of nodes in each layer

example:

3
3
1
2

2. insert truth table to task_data.txt:

example:

F F F F T
F F T T T
F T F F F
F T T T F
T F F T T
T F T T T
T T F T F
T T T T F

3. run:

python main.py
