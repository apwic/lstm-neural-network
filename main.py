from models.Sequential import Sequential
from layers.Dense import DenseLayer
from layers.LSTM import LSTMLayer
from enums.enums import ActivationFunction
from utils.utils import getDataset
import numpy as np

def main():
    input = np.array([
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5]
    ])

    input = getDataset('Test_stock_market.csv')

    model = Sequential()
    model.setInput(input)
    model.add(LSTMLayer(units=64))
    model.add(DenseLayer(units=5, activation_function=ActivationFunction.RELU))
    model.printSummary()
    model.predict()
    model.saveModel('test')

    load = Sequential()
    load.setInput(input)
    load.loadModel('test')
    load.printSummary()
    load.predict()


if __name__ == '__main__':
    main()