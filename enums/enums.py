from enum import Enum

class ActivationFunction(Enum):
    RELU = 0
    SIGMOID = 1
    TANH = 2

class DenseLayerType(Enum):
    HIDDEN = 0
    OUTPUT = 1
