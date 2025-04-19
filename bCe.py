import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
y = [0, 1, 1, 0]
np.random.seed(0)
W1 = [[np.random.rand(), np.random.rand()],
      [np.random.rand(), np.random.rand()]]  
b1 = [np.random.rand(), np.random.rand()]    
W2 = [np.random.rand(), np.random.rand()]   
b2 = np.random.rand()                         
lr = 0.1
epochs = 10000
for _ in range(epochs):
    for i in range(4):
        x0, x1 = X[i]
        t = y[i]
        h0_in = x0 * W1[0][0] + x1 * W1[1][0] + b1[0]
        h1_in = x0 * W1[0][1] + x1 * W1[1][1] + b1[1]
        h0 = sigmoid(h0_in)
        h1 = sigmoid(h1_in)
        o_in = h0 * W2[0] + h1 * W2[1] + b2
        o = sigmoid(o_in)
        error = t - o
        d_o = error * sigmoid_derivative(o)
        d_h0 = d_o * W2[0] * sigmoid_derivative(h0)
        d_h1 = d_o * W2[1] * sigmoid_derivative(h1)
        W2[0] += lr * d_o * h0
        W2[1] += lr * d_o * h1
        b2 += lr * d_o
        W1[0][0] += lr * d_h0 * x0
        W1[1][0] += lr * d_h0 * x1
        b1[0] += lr * d_h0
        W1[0][1] += lr * d_h1 * x0
        W1[1][1] += lr * d_h1 * x1
        b1[1] += lr * d_h1
print("Final XOR Predictions:")
for i in range(4):
    x0, x1 = X[i]
    h0 = sigmoid(x0 * W1[0][0] + x1 * W1[1][0] + b1[0])
    h1 = sigmoid(x0 * W1[0][1] + x1 * W1[1][1] + b1[1])
    o = sigmoid(h0 * W2[0] + h1 * W2[1] + b2)
    print(f"Input: {X[i]} → Output: {round(o)}")