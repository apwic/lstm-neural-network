import numpy as np

def relu( val, deriv=False):
    if (deriv):
        return np.where(val <= 0, 0, 1)
    
    return np.maximum(0, val)

def tanh(val):
    return np.tanh(val)
    
def sigmoid(val, deriv=False):
    val = np.clip(val, -709, 709)  # Clip values to avoid overflow
    if (deriv):
        sigmoid_x = 1 / (1 + np.exp(-val))
        return sigmoid_x * (1 - sigmoid_x)
    return 1 / (1 + np.exp(-val))
