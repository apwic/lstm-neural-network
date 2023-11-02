import numpy as np
from utils.utils import sigmoid, tanh

class LSTMLayer:
    def __init__(self, units):
        self.units = units
        
        self.input: np.ndarray = None

        self.seq_length = None
        self.input_shape = None

        self.output: np.ndarray = None
        self.cell_state: np.ndarray = None

        # xavier (glorot) initialization for weights
        self.bound = None
        
        # weight for phases
        self.Uf = None
        self.Ui = None
        self.Uc = None
        self.Uo = None
        
        # recurrent weights for phases
        self.Wf = None
        self.Wi = None
        self.Wc = None
        self.Wo = None

        # biases for phases
        self.Bf = None
        self.Bi = None
        self.Bc = None
        self.Bo = None

    def __str__(self):
        return f"\nLSTM LAYER\n--------\nInput: {self.input}\n\nOutput: {self.output}\n"

    def __randomU(self):
        return np.random.uniform(-self.bound, self.bound, (self.input_shape, self.units))
    
    def __randomW(self):
        return np.random.uniform(-self.bound, self.bound, (self.units, self.units))

    def __randomB(self):
        return np.zeros((1, self.units))
    
    def setOutput(self, output):
        self.output = output
        
    def setRecurrentWeight(self, wf, wi, wc, wo):
        self.Wf = wf
        self.Wi = wi
        self.Wc = wc
        self.Wo = wo

    def setWeight(self, uf, ui, uc, uo):
        self.Uf = uf
        self.Ui = ui
        self.Uc = uc
        self.Uo = uo

    def setBias(self, bf, bi, bc, bo):
        self.Bf = bf
        self.Bi = bi
        self.Bc = bc
        self.Bo = bo

    def setInputSize(self, dimension):
        self.seq_length = dimension[0]
        self.input_shape = dimension[1]

    def setInput(self, input, isLoad=False):
        self.input = input

        self.seq_length = self.input.shape[0]
        self.input_shape = self.input.shape[1]

        self.output: np.ndarray = np.zeros((1, self.units))
        self.cell_state: np.ndarray = np.zeros((1, self.units))

        if (not isLoad) :
            # xavier (glorot) initialization for weights
            self.bound = np.sqrt(6.0 / (self.input_shape + self.units))
            
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
    
    def getOutput(self):
        return self.output
    
    def getOutputShape(self):
        return (None, self.units)
    
    def getParamsCount(self):
        per_gate = (self.input_shape * self.units) + (self.units * self.units) + self.units
        return per_gate * 4
    
    def getData(self):
        return {
            "type": "lstm",
            "params": {
                "units": self.units,
                # WEIGHT
                "Uf": self.Uf.tolist(),
                "Ui": self.Ui.tolist(),
                "Uc": self.Uc.tolist(),
                "Uo": self.Uo.tolist(),
                # RECURRENT WEIGHT
                "Wf": self.Wf.tolist(),
                "Wi": self.Wi.tolist(),
                "Wc": self.Wc.tolist(),
                "Wo": self.Wo.tolist(),
                # BIASES
                "Bf": self.Bf.tolist(),
                "Bi": self.Bi.tolist(),
                "Bc": self.Bc.tolist(),
                "Bo": self.Bo.tolist(),
            }
        }

    def forgetGate(self, input):
        self.cell_state = self.cell_state * (sigmoid(np.matmul(input, self.Uf) + np.matmul(self.output, self.Wf) + self.Bf))
    
    def inputGate(self, input):
        it = sigmoid(np.matmul(input, self.Ui) + np.matmul(self.output, self.Wf) + self.Bi)
        ct = tanh(np.matmul(input, self.Uc) + np.matmul(self.output, self.Wc) + self.Bc)

        self.cell_state = self.cell_state + (it * ct)

    def outputGate(self, input):
        ot = sigmoid(np.matmul(input, self.Uo) + np.matmul(self.output, self.Wo) + self.Bo)

        self.output = ot * tanh(self.cell_state)

    def forward(self):
        for i in range(self.seq_length):
            self.forgetGate(self.input[i])
            self.inputGate(self.input[i])
            self.outputGate(self.input[i])
