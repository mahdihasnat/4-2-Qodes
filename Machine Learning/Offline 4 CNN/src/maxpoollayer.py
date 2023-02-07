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
        self.x_shape = x.shape
        out_shape =( (x.shape[2] - self.kernel_shape[0])//self.stride[0] + 1,\
                     (x.shape[3] - self.kernel_shape[1])//self.stride[1] + 1)
        
        strided_x = as_strided(x, 
            strides=(x.strides[0], x.strides[1] , x.strides[2] * self.stride[0], 
                     x.strides[3] * self.stride[1] , x.strides[2] , x.strides[3] ),
            shape = (x.shape[0], x.shape[1], out_shape[0], out_shape[1], self.kernel_shape[0], self.kernel_shape[1])
                        )
        reshaped_strided_x = strided_x.reshape(x.shape[0],x.shape[1], out_shape[0], out_shape[1], -1)
        out_arg = np.argmax(reshaped_strided_x,axis = 4)
        self.out_arg=out_arg
        # print("out_arg shape: ",out_arg.shape)
        # print("out_arg: ",out_arg)
        out_x = strided_x.max(axis=(4,5))
        assert out_x.shape == (x.shape[0], x.shape[1], out_shape[0], out_shape[1])
        return out_x

    def backward(self, del_z, lr):
        """
        in del_z: (batch_size, in_channels, out_height, out_width)
        out del_x: (batch_size, in_channls, height, width)
        """
        assert len(del_z.shape) == 4 , "del_z is not 4D"
        del_x = np.zeros(self.x_shape)
        print("self.out_args: ",self.out_arg)
        p = np.unravel_index(self.out_arg,shape=self.kernel_shape)
        print("p:",p)
        # print("p shape:",p.shape)
        print("del_z shape: ",del_z.shape)
        
        del_x[:,:,p]+=del_z
        print(del_x[:,:,p])
        
        

if __name__ == '__main__':
    x_shape = (1,1,3,3)
    k_shape = (2,2)
    stride = (1,1)
    x = np.array([1,3,2,
                  4,-2,-5,
                  -5,-6,-1])
    x = x.reshape(x_shape)
    print("X shape: ",x.shape)
    assert x.shape == x_shape ,"shape dont match"
    maxpoollayer = MaxPool2d(k_shape,stride)
    a = maxpoollayer.forward(x)
    print("a:",a)
    b = maxpoollayer.backward(a,0.1)