from enums.enums import ActivationFunction, DenseLayerType
from utils.utils import relu, sigmoid
import numpy as np

class DenseLayer:
    def __init__(
        self,
        units: float,
        activation_function: ActivationFunction,
    ) -> None:
        self.units = units
        self.activation_function = activation_function
        self.input_size: None
        self.input: np.ndarray = []
        self.output: np.ndarray = []
        self.weights: np.ndarray = []
        self.biases = np.zeros((1, units))
    
    def __str__(self):
        return f"\nDENSE LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"
    
    def setInputSize(self, input_size):
        self.input_size = input_size

    def setInput(self, input: np.ndarray):
        self.input = input.ravel()
        self.input_size = len(self.input)
        self.delta_w = np.zeros((len(input.ravel()), self.units))

        if (len(self.weights) == 0):
            self.weights = np.random.randn(len(input.ravel()), self.units) * np.sqrt(2. / len(input.ravel()))

    def setWeight(self, weights: np.ndarray):
        self.weights = weights

    def setBiases(self, biases: np.ndarray):
        self.biases = biases

    def getOutput(self):
        return self.output
    
    def getOutputShape(self):
        return (1, self.units)
    
    def getParamsCount(self):
        return self.input_size * self.units
    
    def getData(self):
        return {
            "type": "dense",
            "params": {
                "units": self.units,
                "activation_function": str(self.activation_function),
                "learning_rate": self.learning_rate,
                "kernel": self.weights.tolist(),
                "biases": self.biases.tolist()
            }
        }

    def forward(self):
        z = np.dot(self.input, self.weights) + self.biases
        self.net = z

        if (self.activation_function == ActivationFunction.RELU):
            self.output = relu(z)
        elif (self.activation_function == ActivationFunction.SIGMOID):
            self.output = sigmoid(z)

