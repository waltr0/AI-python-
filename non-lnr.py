import numpy as np
def tanh(x):
    return np.tanh(x)
input_value=np.array([-10,3,5,10])
output_value=tanh(input_value)
print("input: ",input_value)
print("tanh output: ",output_value)

def relu(y):
    return np.maximum(0,y)
output1_value=relu(input_value)
print("ReLu Output: ",output1_value)

def sigmoid(z):
    return 1/(1+np.exp(-z))
output2_value=sigmoid(input_value)
print("sigmoid output: ",output2_value)