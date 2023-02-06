import numpy as np

class SoftMax():
    
    def forward(self,x):
        """
            in x: (samples, classes)
               out x: (samples, classes)
        """
        assert len(x.shape) == 2, "The input of SoftMax must be 2-D"
        exp = np.exp(x)
        return exp / np.sum( exp , axis = 1, keepdims = True)
    


if __name__ == '__main__':
    x_shape = (100,10)
    x = np.random.randint(-100,100,x_shape)
    print(x.shape)
    s = SoftMax()
    z = s.forward(x)
    print(z.shape)
    