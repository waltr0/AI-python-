import numpy as np
X = np.array([1, 2, 3, 4, 5])
y = np.array([1.2, 1.9, 3.0, 3.8, 5.1])
w, b = 0, 0  # Start with 0
learning_rate = 0.01
iterations = 10
m = len(y)
for _ in range(iterations):
    y_pred = w * X + b
    errors = y_pred - y
    dw = (1 / m) * np.dot(errors, X)
    db = (1 / m) * np.sum(errors)
    w -= learning_rate * dw
    b -= learning_rate * db
print(f"Weight (w): {w}, Bias (b): {b}")