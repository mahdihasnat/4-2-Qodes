import numpy as np

class CNN():
    
    def __init__(self) -> None:
        self.layers = []
        
    
    def add_layer(self,layer):
        self.layers.append(layer)
    

    def train(self,x,y_true,lr):
        for layer in self.layers:
            x = layer.forward(x)
            print("train output shape: ",x.shape)
        
    
    

