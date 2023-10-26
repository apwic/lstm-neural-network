import numpy as np

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
