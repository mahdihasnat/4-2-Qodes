import numpy as np


class ReLU():
    
    def __init__(self) -> None:
        """
            https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        """
        

    def forward(self, x):
        """
            x: shape = (batch_size, channels, height, width)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        self.x = x
        return np.maximum(x,0)
        
    def backward(self, del_z, lr):
        """
            del_z: shape = (batch_size, channels, height, width)
        """
        assert len(del_z.shape) == 4, "input shape is not 4D"
        assert del_z.shape == self.x.shape, "del_z shape is not same as x shape"
        
        return del_z * (self.x >= 0)


if __name__ == '__main__':
    # test
    x = np.array([[[[1,-2,3],[4,5,6],[7,8,9]]]])
    relu = ReLU(inplace = True)
    y = relu.forward(x)
    print(y)