from models.Sequential import Sequential
from layers.Dense import DenseLayer
from layers.LSTM import LSTMLayer
from enums.enums import ActivationFunction
import numpy as np

def main():
    input = np.array([
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5]
    ])

    model = Sequential()
    model.setInput(input)
    model.add(LSTMLayer(hidden_units=64))
    model.add(DenseLayer(units=5, activation_function=ActivationFunction.RELU))
    model.printSummary()
    model.predict()

if __name__ == '__main__':
    main()