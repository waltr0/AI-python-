import numpy as np
def step_function(X):
    return 1 if X >= 0 else 0
def perceptron(X1, X2, weights, bias):
    weighted_sum = weights[0] * X1 + weights[1] * X2 + bias
    return step_function(weighted_sum)
weights = np.array([1, 1])
bias = -1.5
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
for X1, X2 in inputs:
    output = perceptron(X1, X2, weights, bias)
    print(f"Input: ({X1}, {X2}) -> Output: {output}")