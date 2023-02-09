import numpy as np
from numpy.lib.stride_tricks import as_strided
from fast_convulation import fast_convulate, fast_hadamard, fast_hadamard_weight_calc,fast_convulate_x_calc
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
        
        # local variables
        self.x_shape  = None
        self.padded_x = None
    
    
    def forward(self,x):
        """
            x: shape = (batch_size, in_channels, height, width)
            out_x : shape = (batch_size, out_channels, out_height, out_width)
            weights shape = (out_channels, in_channels, kernel_shape[0], kernel_shape[1])
            
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        in_channels = x.shape[1]
        batch_size = x.shape[0]
        self.x_shape = x.shape
        
        
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
        self.padded_x = padded_x
        out_x = fast_hadamard(padded_x,self.weights)
        out_x = out_x[:,:,::self.stride[0],::self.stride[1]]
        
        self.biases = self.biases.reshape((1,self.out_channels,1,1))
        out_x = np.add(out_x,self.biases)
        self.biases = self.biases.reshape((self.out_channels,))
        
        assert out_x.shape == (batch_size,self.out_channels,out_shape[0],out_shape[1])
        return out_x
    
    
    def backward(self, del_z, lr):
        """
            in del_z: (batch_size, out_channels, out_height, out_width)
            out del_x: (batch_size, in_channels, in_height, in_width)
        """
        assert len(del_z.shape) == 4, "input shape is not 4D"
        assert del_z.shape[1] == self.out_channels, "output channel dont match"
        
        batch_size = self.x_shape[0]
        in_channels = self.x_shape[1]
        
        modified_del_z = np.zeros((batch_size,self.out_channels,
                                self.stride[0] * (del_z.shape[2] - 1) +1,
                                self.stride[1] * (del_z.shape[3] - 1) +1))

        modified_del_z[:,:,::self.stride[0],::self.stride[1]] = del_z
        
        # print("self.padded_x.shape ",self.padded_x.shape)
        # print("modified_del_z.shape: ",modified_del_z.shape)
        del_w = fast_hadamard_weight_calc(self.padded_x,modified_del_z)
        del_w = del_w[:,:,:self.kernel_shape[0]:,:self.kernel_shape[1]:]
        del_w /= batch_size
        # print("del_w.shape: ",del_w.shape)
        assert del_w.shape == self.weights.shape, "del_w shape dont match"
        
        # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
        extra_end_padding = (self.padded_x.shape[2] - modified_del_z.shape[2],
                             self.padded_x.shape[3] - modified_del_z.shape[3])
 
        padded_modified_del_z_size = (modified_del_z.shape[2] + self.kernel_shape[0]-1 + extra_end_padding[0],
                                      modified_del_z.shape[3] + self.kernel_shape[1]-1 + extra_end_padding[1])
        padded_modified_del_z = np.pad(modified_del_z, ((0,0),
                                                        (0,0),
                                                        (self.kernel_shape[0]-1,extra_end_padding[0]),
                                                        (self.kernel_shape[1]-1,extra_end_padding[1])))
        # padded_modified_del_z shape = (batch_size, out_channel, height,width)
        # weights shape = (out_channel, in_channel, kernel_height, kernel_width)
        
        del_padded_x = fast_convulate_x_calc(padded_modified_del_z,self.weights)
        assert del_padded_x.shape == self.padded_x.shape, "del_padded_x shape dont match"
        
        del_x = del_padded_x[:,:,
                             self.padding[0]:self.padded_x.shape[-2]-self.padding[0]:,
                             self.padding[1]:self.padded_x.shape[-1]-self.padding[1]:]
        assert del_x.shape == self.x_shape, "del_x shape dont match"
        
        # del_z: shape = (batch_size, out_channel, out_height, out_width)
        # del_b: shape = (out_channel,)
        del_b = np.sum(del_z,axis=(0,2,3))/batch_size
        assert del_b.shape == self.biases.shape, "del_b shape dont match"
        
        self.weights -= lr*del_w
        self.biases -= lr*del_b
        
        return del_x

    def clean(self):
        self.x_shape  = None
        self.padded_x = None
        

if __name__ == '__main__':
    np.random.seed(0)
    while True:
        lim = 10
        stride = (np.random.randint(1,lim),np.random.randint(1,lim))
        padding = (np.random.randint(0,lim),np.random.randint(0,lim))
        out_channels = np.random.randint(1,lim)
        kernel_shape = (np.random.randint(1,lim),np.random.randint(1,lim))
        x_shape = (np.random.randint(1,lim),np.random.randint(1,lim),np.random.randint(1,lim),np.random.randint(1,lim))
        if x_shape[2] < kernel_shape[0] or x_shape[3] < kernel_shape[1]:
            continue
        my = Conv2d( out_channels=out_channels, kernel_size=kernel_shape, stride=stride,padding=padding)
        x = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9]).reshape(1,1,3,3)
        x = np.zeros(x_shape)
        print("x: ",x)
        print("x.shape: ",x.shape)
        z = my.forward(x)
        print("z: ",z)
        my.backward(z,0.1)
    
    