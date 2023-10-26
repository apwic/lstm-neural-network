import os
import numpy as np
import pandas as pd

base_dir = 'dataset'

def relu( val, deriv=False):
    if (deriv):
        return np.where(val <= 0, 0, 1)
    
    return np.maximum(0, val)

def tanh(val):
    return np.tanh(val)
    
def sigmoid(val, deriv=False):
    val = np.clip(val, -709, 709)  # clip values to avoid overflow
    if (deriv):
        sigmoid_x = 1 / (1 + np.exp(-val))
        return sigmoid_x * (1 - sigmoid_x)
    return 1 / (1 + np.exp(-val))

def getDataset(csv_file):
    csv_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 1 :-1].values  # cuma butuh kolom ke 2 - 6

    return data
