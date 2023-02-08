import numpy as np


class ReLU():
    
    def __init__(self) -> None:
        """
            https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        """
        self.x = None

    def forward(self, x):
        """
            x: shape = (batch_size, ...)
        """
        assert len(x.shape)>=2, "input shape is not at least 2D"
        self.x = x
        return np.maximum(x,0)
        
    def backward(self, del_z, lr):
        """
            del_z: shape = (batch_size, ...)
        """
        assert len(del_z.shape)>=2, "input shape is not at least 2D"
        assert del_z.shape == self.x.shape, "del_z shape dont match"
        
        return del_z * (self.x >= 0)

    def clean(self):
        self.x = None
    

if __name__ == '__main__':
    # test
    x = np.array([[[[1,-2,3],[4,5,6],[7,8,9]]]])
    relu = ReLU(inplace = True)
    y = relu.forward(x)
    print(y)