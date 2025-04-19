import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.linspace(0, 10, 100)
Y = 2 * X + np.random.normal(0, 1, size=X.shape)
X_b = np.c_[np.ones(X.shape[0]), X] 
weights = np.zeros(X_b.shape[1])
learning_rate = 0.01
epochs = 100
for epoch in range(epochs):
    for i in range(X_b.shape[0]):
        xi = X_b[i]
        yi = Y[i]
        prediction = np.dot(xi, weights)
        error = yi - prediction
        weights += learning_rate * error * xi
print("Final weights (bias, slope):", weights)
Y_pred = X_b @ weights
plt.scatter(X, Y, color='black', label='Actual data')
plt.plot(X, Y_pred, color='red', label='LMS Prediction')
plt.title("LMS Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()