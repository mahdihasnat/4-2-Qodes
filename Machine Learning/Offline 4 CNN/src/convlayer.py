import numpy as np
from numpy.lib.stride_tricks import as_strided
from fast_convulation import fast_convulate
# cnn convolutional layer
class Conv2d:
    
    def __init__(self, out_channels, kernel_size, stride=1, padding=0) -> None:
        """
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
        
        self.out_channels = out_channels
        self.kernel_shape = kernel_size
        self.stride = stride
        self.padding = padding
        
        # print("kernel_shape: ",self.kernel_shape)
        # print("stride: ",self.stride)
        # print("padding: ",self.padding)
        
        self.weights = None
        self.biases = None
    
    
    def forward(self,x):
        """
            x: shape = (batch_size, in_channels, height, width)
            out_x : shape = (batch_size, out_channels, out_height, out_width)
            weights shape = (out_channels, in_channels, kernel_shape[0], kernel_shape[1])
            
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        in_channels = x.shape[1]
        batch_size = x.shape[0]
        
        if self.weights is None:
            # He initialization for RElu
            # https://paperswithcode.com/method/he-initialization
            # weight shape = (out_channels, in_channels, kernel_shape[0], kernel_shape[1])
            self.weights = np.random.randn(self.out_channels, in_channels, self.kernel_shape[0], self.kernel_shape[1])  \
                            * np.sqrt(2/(in_channels*self.kernel_shape[0]*self.kernel_shape[1]))

        if self.biases is None:
            # bias shape = (out_channels)
            self.biases = np.random.randn(self.out_channels)\
                    *np.sqrt(2/(in_channels*self.kernel_shape[0]*self.kernel_shape[1]))
        
        assert self.weights.shape == (self.out_channels, in_channels, self.kernel_shape[0], self.kernel_shape[1]) , "wight shape dont match"
        assert self.biases.shape == (self.out_channels,) , "bias shape dont match"
        
        out_shape =( (x.shape[2] + 2*self.padding[0] - self.kernel_shape[0])//self.stride[0] + 1,\
                     (x.shape[3] + 2*self.padding[1] - self.kernel_shape[1])//self.stride[1] + 1)
        
        assert out_shape[0] > 0 and out_shape[1] > 0, "output shape is negative"
        
        # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        padded_x = np.pad(x , ((0,0),(0,0),(self.padding[0],self.padding[0]),(self.padding[1],self.padding[1])),\
                            mode='constant', constant_values=0)
        out_x = fast_convulate(padded_x,self.weights)
        out_x = out_x[:,:,::self.stride[0],::self.stride[1]]
        
        self.biases = self.biases.reshape((1,self.out_channels,1,1))
        out_x = np.add(out_x,self.biases)
        self.biases = self.biases.reshape((self.out_channels,))
        
        assert out_x.shape == (batch_size,self.out_channels,out_shape[0],out_shape[1])
        return out_x


if __name__ == '__main__':
    
    my = Conv2d( 2, 1, stride=1,padding=(1,1))
    x = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9]).reshape(1,1,3,3)
    print("x: ",x)
    print("x.shape: ",x.shape)
    z = my.forward(x)
    print("z: ",z)
    
    