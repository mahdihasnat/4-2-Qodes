import numpy as np
from numpy.lib.stride_tricks import as_strided

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
        assert self.kernel_shape[0] > 0 and self.kernel_shape[1] > 0, "kernel size must be positive"
        assert self.stride[0] > 0 and self.stride[1] > 0, "stride must be positive"
        
    def forward(self,x):
        """
            input shape = (batch_size, in_channels, height, width)
            output shape = (batch_size, in_channels, out_height, out_width)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        
        out_shape =( (x.shape[2] - self.kernel_shape[0])//self.stride[0] + 1,\
                     (x.shape[3] - self.kernel_shape[1])//self.stride[1] + 1)
        
        strided_x = as_strided(x, 
            strides=(x.strides[0], x.strides[1] , x.strides[2] * self.stride[0], 
                     x.strides[3] * self.stride[1] , x.strides[2] , x.strides[3] ),
            shape = (x.shape[0], x.shape[1], out_shape[0], out_shape[1], self.kernel_shape[0], self.kernel_shape[1])
                        )
        
        out_x = strided_x.max(axis=(4,5))
        assert out_x.shape == (x.shape[0], x.shape[1], out_shape[0], out_shape[1])
        return out_x


