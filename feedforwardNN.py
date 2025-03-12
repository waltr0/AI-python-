import numpy as np
import pandas as pd
df = pd.DataFrame([[8, 8, 4], [7, 9, 5], [6, 10, 6], [5, 12, 7]], columns=['cgpa', 'profile_score', 'lpa'])
def initialize_parameters(layer_dims):
    np.random.seed(3)
    print("Layer dimensions:", layer_dims)
    parameters = {}
    L = len(layer_dims)
    print("Total number of layers in neural network:", L)
    for i in range(1, L):
        parameters['w' + str(i)] = np.ones((layer_dims[i - 1], layer_dims[i])) * 0.1
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    return parameters
def linear_forward(A_prev, w, b):
    z = np.dot(w.T, A_prev) + b
    print(z)
    return z
def relu(z):
    return np.maximum(0, z)
def L_layer_forward(X, parameters):
    A = X
    caches = []
    L = len(parameters) // 2 
    for i in range(1, L):
        A_prev = A
        w = parameters['w' + str(i)]
        b = parameters['b' + str(i)]
        z = linear_forward(A_prev, w, b)
        A = relu(z)
        cache = (A_prev, w, b, z)
        caches.append(cache)
    w_out = parameters['w' + str(L)]
    b_out = parameters['b' + str(L)]
    z_out = linear_forward(A, w_out, b_out)
    AL = z_out
    return AL, caches
X = df[['cgpa', 'profile_score']].values[0].reshape(2, 1)
parameters = initialize_parameters([2, 2, 1])
y_hat, caches = L_layer_forward(X, parameters)
print("Final output")
print(y_hat)