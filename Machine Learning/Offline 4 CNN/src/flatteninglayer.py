import numpy as np


class FlatteningLayer():
    
    def __init__(self) -> None:
        pass
    
    def forward(self,x):
        """
			in x: (batch_size, ...)
			out x: (batch_size, ...)
        """
        assert len(x.shape) >= 2, "input shape is not at least 2D"
        self.x_shape = x.shape
        return x.reshape(x.shape[0],-1)
    
    def backward(self,del_z,lr):
        """
            in del_z: (batch_size, ...)
        """
        return del_z.reshape(self.x_shape)

    def clean(self):
        pass

if __name__ == '__main__':
    x_shape = (100,15,32,32)
    x = np.random.randint(-100,100,x_shape)
    f = FlatteningLayer()
    y = f.forward(x)
    print(y.shape)