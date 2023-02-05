import numpy as np


# cnn convolutional layer
class ConvLayer:
    
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        """
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: size of the convolutional kernel, int or pair
        stride: stride of the convolution
        padding: padding of the input
        """
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        assert len(kernel_size) == 2
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        pass
    
    
    



import torch.nn as nn
import torch
if __name__ == '__main__':
    # Test for My convolutional layer using Pytorch convolutional layer
    
    m = nn.Conv2d(1, 1, 3, stride=1,dtype=torch.float64)
    my = ConvLayer(1, 1, 3, stride=1)
    x = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9]).reshape(1,1,3,3)
    print("x: ",x)
    print("x.shape: ",x.shape)
    xx = torch.from_numpy(x)
    y = m(xx)
    print("y: ",y)
    
    