import numpy as np
import json
from enums.enums import ActivationFunction
from layers.LSTM import LSTMLayer
from layers.Dense import DenseLayer

class Sequential():
    def __init__(self):
        self.layers = []
        self.input: np.ndarray = []
        self.output: np.ndarray = []

    def __isLSTM(self, cls):
        return cls.__class__.__name__ == "LSTMLayer"
    
    def __isDense(self, cls):
        return cls.__class__.__name__ == "DenseLayer"

    def print_separator(self, title=None, width=79):
        print("=" * width)
        if title:
            print(title.center(width))
        print("=" * width)

    def printSummary(self):
        print("———————————————————————————————————————————————————————————————————————")
        print("{:<30} {:<30} {:<10}".format(
            'Layer (type) ', 'Output Shape', 'Param #'))
        print("=======================================================================")
        sum_parameter = 0
        lstm_count = 0
        dense_count = 0

        for layer in self.layers:
            if (self.__isLSTM(layer)):
                if (lstm_count == 0):
                    postfix = " (LSTM)"
                else:
                    postfix = "_" + str(lstm_count) + " (LSTM)"
                lstm_count += 1

                layerTypes = 'lstm' + postfix

            if (self.__isDense(layer)):
                if (dense_count == 0):
                    postfix = " (Dense)"
                else:
                    postfix = "_" + str(dense_count) + " (Dense)"
                dense_count += 1

                layerTypes = 'dense' + postfix

            shape = layer.getOutputShape()
            weight = layer.getParamsCount()
            sum_parameter += weight
            print("{:<30} {:<30} {:<10}".format(
                layerTypes, str(shape), weight))
            if (layer != self.layers[len(self.layers)-1]):
                print(
                    "———————————————————————————————————————————————————————————————————————")
            else:
                print(
                    "=======================================================================")

        trainable_parameter = sum_parameter
        non_trainable_parameter = sum_parameter - trainable_parameter

        print("Total Params: {}".format(sum_parameter))
        print("Trainable Params: {}".format(trainable_parameter))
        print("Non-trainable Params: {}".format(non_trainable_parameter))
        print()

    def saveModel(self, filename):
        file = open(f'./output/{filename}.json' , 'w')
        data = []

        for layer in self.layers:
            data.append(layer.getData())

        file.write(json.dumps(data, indent=4))
        file.close()
        print("MODEL SAVED")

    def loadModel(self, filename):
        file = open(f'./output/{filename}.json', 'r')

        data = json.load(file)
        file.close()

        self.layers = []

        # loop through the loaded data and reconstruct the layers
        for layer_data in data:
            layer_type = layer_data["type"]
            
            # Add LSTM layer
            if layer_type == "lstm":
                params = layer_data["params"]

                lstmLayer = LSTMLayer(
                    units=params["units"]
                )

                lstmLayer.setWeight(params["Uf"], params["Ui"], params["Uc"], params["Uo"])
                lstmLayer.setRecurrentWeight(params["Wf"], params["Wi"], params["Wc"], params["Wo"])
                lstmLayer.setBias(params["Bf"], params["Bi"], params["Bc"], params["Bo"])
                self.add(lstmLayer)
            
            # Add dense layer
            elif layer_type == "dense":
                params = layer_data["params"]
                activation_function = ActivationFunction.RELU

                if (params["activation_function"] == "ActivationFunction.RELU"):
                    activation_function = ActivationFunction.RELU
                if (params["activation_function"] == "ActivationFunction.SIGMOID"):
                    activation_function = ActivationFunction.SIGMOID

                denseLayer = DenseLayer(
                    units=params["units"],
                    activation_function=activation_function,
                )
                denseLayer.setWeight(np.array(params["kernel"]))
                denseLayer.setBiases(np.array(params["biases"]))
                self.add(denseLayer)
            
        print("MODEL LOADED")

    def setInput(self, input):
        self.input = input

    def getOutput(self):
        return self.output

    def add(self, layer):
        if (self.__isLSTM(layer)):
            layer.setInput(self.input)
        else:
            layer.setInputSize(self.layers[-1].getOutputShape()[1])

        self.layers.append(layer)

    def predict(self):
        i = 0
        for layer in self.layers:
            if (self.__isDense(layer)):
                layer.setInput(self.output)
            
            # if layer is LSTM but not first LSTM
            if (i != 0 and self.__isLSTM(layer)):
                layer.setOuput(self.output)

            layer.forward()
            self.output = layer.getOutput()
            print(layer)
            i += 1
