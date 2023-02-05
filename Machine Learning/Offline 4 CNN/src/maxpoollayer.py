import numpy as np

class MaxPool2d():
    
    def __init__(self, kernel_size, stride):
        """
        kernel_size: size of the convolutional kernel, int or pair
        stride: stride of the convolution
        """
        if type(kernel_size) == int:
            self.kernel_shape = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2
            assert type(kernel_size) == tuple
            self.kernel_shape = kernel_size
        if type(stride) == int:
            self.stride = (stride, stride)
        else:
            assert len(stride) == 2
            assert type(stride) == tuple
            self.stride = stride
        
    def forward(self,x):
        pass