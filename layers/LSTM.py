import numpy as np
from utils.utils import sigmoid, tanh

class LSTMLayer:
    # LSTMLayer(dimension = (4, 5))
    def __init__(self, dimension, hidden_units):
        self.seq_length = dimension[0]
        self.input_shape = dimension[1]
        self.hidden_units = hidden_units
        
        self.input: np.ndarray = None
        self.output: np.ndarray = np.zeros((1, self.hidden_units))
        self.cell_state: np.ndarray = np.zeros((1, self.hidden_units))

        # xavier (glorot) initialization for weights
        self.bound = np.sqrt(6.0 / (self.input_shape + self.hidden_units))
        
        # weight for phases
        self.Uf = self.__randomU()
        self.Ui = self.__randomU()
        self.Uc = self.__randomU()
        self.Uo = self.__randomU()
        
        # recurrent weights for phases
        self.Wf = self.__randomW()
        self.Wi = self.__randomW()
        self.Wc = self.__randomW()
        self.Wo = self.__randomW()

        # biases for phases
        self.Bf = self.__randomB()
        self.Bi = self.__randomB()
        self.Bc = self.__randomB()
        self.Bo = self.__randomB()

    def __str__(self):
        pass

    def __randomU(self):
        return np.random.uniform(-self.bound, self.bound, (self.input_shape, self.hidden_units))
    
    def __randomW(self):
        return np.random.uniform(-self.bound, self.bound, (self.hidden_units, self.hidden_units))

    def __randomB(self):
        return np.zeros((1, self.hidden_units))
    
    def getOutput(self):
        return self.output

    def forgetGate(self):
        self.cell_state = self.cell_state * (sigmoid((self.Uf * self.input) + (self.Wf * self.output) + self.Bf))
    
    def inputGate(self):
        it = sigmoid((self.Ui * self.input) + (self.Wi * self.output) + self.Bi)
        ct = tanh((self.Uc * self.input) + (self.Wc * self.output) + self.Bc)

        self.cell_state = self.cell_state + (it * ct)

    def outputGate(self):
        ot = sigmoid((self.Uo * self.input) + (self.Wo * self.output) + self.Bo)

        self.output = ot * tanh(self.cell_state)

    def forward(self):
        for i in range(self.seq_length):
            self.forgetGate()
            self.inputGate()
            self.outputGate()
