import numpy as np


class ReLU():
    
    def __init__(self,inplace = False) -> None:
        """
            https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        """
        self.inplace = inplace

    def forward(self, x):
        """
            x: shape = (batch_size, channels, height, width)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        if self.inplace:
            np.maximum(x,0,x)
            return x
        else:
            return np.maximum(x,0)
        


if __name__ == '__main__':
    # test
    x = np.array([[[[1,-2,3],[4,5,6],[7,8,9]]]])
    relu = ReLU(inplace = True)
    y = relu.forward(x)
    print(y)