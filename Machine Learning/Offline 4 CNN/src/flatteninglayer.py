import numpy as np


class FlatteningLayer():
    
    def __init__(self) -> None:
        pass
    
    def forward(self,x):
        """
			in x: (batch_size, channels, height, width)
			out x: (batch_size, channels*height*width)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        return x.reshape(x.shape[0],-1)
    

if __name__ == '__main__':
    x_shape = (100,15,32,32)
    x = np.random.randint(-100,100,x_shape)
    f = FlatteningLayer()
    y = f.forward(x)
    print(y.shape)