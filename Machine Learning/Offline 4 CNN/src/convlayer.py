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
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        assert len(kernel_size) == 2
        
        if type(stride) == int:
            stride = (stride, stride)
        assert len(stride) == 2
        
        if type(padding) == int:
            padding = (padding, padding)
        assert len(padding) == 2
        
        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel_size
        self.stride = stride
        self.padding = padding
        
        print("kernel_shape: ",self.kernel_shape)
        print("stride: ",self.stride)
        print("padding: ",self.padding)
        
        # He initialization for RElu
        # https://paperswithcode.com/method/he-initialization
        # weight shape = (out_channels, in_channels, kernel_shape[0], kernel_shape[1])
        self.weights = np.random.randn(out_channels, in_channels, self.kernel_shape[0], self.kernel_shape[1])  \
                        * np.sqrt(2/(in_channels*self.kernel_shape[0]*self.kernel_shape[1]))

        
        # bias shape = (out_channels)
        self.bias = np.random.randn(out_channels)\
                *np.sqrt(2/(in_channels*self.kernel_shape[0]*self.kernel_shape[1]))
        
        assert self.weights.shape == (out_channels, in_channels, self.kernel_shape[0], self.kernel_shape[1]) , "wight shape dont match"
        assert self.bias.shape == (out_channels,) , "bias shape dont match"
    
    
    def forward(self,x):
        """
            x: shape = (batch_size, in_channels, height, width) or (in_channels, height, width)
            
        """
        x_shape = x.shape[-3:]
        print("x_shape: ",x_shape)
        out_shape =( (x_shape[1] + 2*self.padding[0] - self.kernel_shape[0] +1)//self.stride[0],\
                    (x_shape[2] + 2*self.padding[1] - self.kernel_shape[1] +1)//self.stride[1] )
        print("out_shape: ",out_shape)
        
        assert out_shape[0] > 0 and out_shape[1] > 0, "output shape is negative"
        
        
        


import torch.nn as nn
import torch
if __name__ == '__main__':
    # Test for My convolutional layer using Pytorch convolutional layer
    
    m = nn.Conv2d(1, 2, 2, stride=1,dtype=torch.float64)
    my = ConvLayer(1, 2, 2, stride=1,padding=0)
    x = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9]).reshape(1,1,3,3)
    print("x: ",x)
    print("x.shape: ",x.shape)
    # xx = torch.from_numpy(x)
    # y = m(xx)
    # print("y: ",y)
    
    z = my.forward(x)
    print("z: ",z)
    
    