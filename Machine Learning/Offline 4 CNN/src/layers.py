import numpy as np

# generic CNN layers 

class Layer():
    
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError
