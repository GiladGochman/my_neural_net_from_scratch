## How to Run:

1. insert neural network dimetions to `network_config.txt` as such:

first row is the number of layers

rest of rows are the number of nodes in each layer

example:

```
3
3
1
2
```

2. insert truth table to `task_data.txt`:

example:

```
F F F F T
F F T T T
F T F F F
F T T T F
T F F T T
T F T T T
T T F T F
T T T T F
```

3. run:

```
python main.py
```

4. after running and diverging successfully:

you may run `predict.py` to perdict a single line of inputs..

    Usage: python predict.py <input1> <input2> ...

    Example: python predict.py T F T
