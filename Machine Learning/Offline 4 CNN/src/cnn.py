import numpy as np

class CNN():
    
    def __init__(self) -> None:
        self.layers = []
        
    
    def add_layer(self,layer):
        self.layers.append(layer)
        
    def cross_entrpy_loss(self,y_pred,y_true):
        # print("y_pred:",y_pred)
        return np.sum(-1 * np.sum(y_true * np.log(y_pred), axis=0))
    
    def train(self,x,y_true,lr):
        """
            in: x shape = (batch_size, channels, height, width)
            in: y_true shape = (batch_size, classes)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        assert len(y_true.shape) == 2, "y_true shape is not 2D"
        for layer in self.layers:
            x = layer.forward(x)
            # print("train output shape: ",x.shape)
        y = x
        # print("shape of y_true",y_true.shape)
        # print("train output:",y)
        print(f"Cross entropy loss: {self.cross_entrpy_loss(y,y_true)}")
        
        del_z = y - y_true
        # print("shape of del_z",del_z.shape)
        for layer in reversed(self.layers):
            del_z = layer.backward(del_z,lr)
            # print("train del_z shape: ",del_z.shape)
    
    def predict(self,x):
        assert len(x.shape) == 4, "input shape is not 4D"
        for layer in self.layers:
            x= layer.forward(x)
        return x

