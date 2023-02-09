import numpy as np
import tqdm

from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.lib.stride_tricks import as_strided


from matplotlib import pyplot as plt
import tqdm
from sklearn import metrics as skm
import numpy as np
import cv2
import pickle as pk
import pandas as pd


def fast_convulate(x,y):
    """
    x: shape = (batch_size, in_channels, height, width)
    y: shape = (out_channels,in_channels, kernel_height, kernel_height)
    out : shape = (batch_size, out_channels, height, width)
    """
    assert len(x.shape) == 4, "x not in 4D"
    assert len(y.shape) == 4, "y not in 4D"
    assert x.shape[1] == y.shape[1], "in_channels dont match"
    n1,m1 = x.shape[-2:]
    n2,m2 = y.shape[-2:]
    n = n1+n2-1
    m = m1+m2-1
    x = np.pad(x,((0,0),(0,0),(0,n-n1),(0,m-m1)),mode = 'constant',constant_values = 0)
    y = np.pad(y,((0,0),(0,0),(0,n-n2),(0,m-m2)),mode = 'constant',constant_values = 0)
    # print("x:",x)
    # print("y:",y)
    fx = fft2(x,axes = (-2,-1))
    fy = fft2(y,axes = (-2,-1))
    assert fx.shape[-2:] == (n,m) , "fx size dont match"
    # print("fx: ",fx.shape)
    # print("fy: ",fy.shape)
    fz = np.einsum("ijkl,mjkl->imkl",fx,fy)
    z = np.real(ifft2(fz,axes = (-2,-1)))
    # print("z.shape: ",z.shape)
    # print("z: ",z)
    # z = np.roll(z,n2-1,axis = -2)
    # z = np.roll(z,m2-1,axis = -1)
    
    assert z.shape == (x.shape[0], y.shape[0], n,m), "z shape dont match"
    z = z[:,:,n2-1:n1,m2-1:m1]
    assert z.shape == (x.shape[0], y.shape[0], n1-n2+1, m1-m2+1) , "z shape dont match"
    return z


def fast_hadamard(x,y):
    """
    x: shape = (batch_size, in_channels, height, width)
    y: shape = (out_channels,in_channels, kernel_height, kernel_width)
    out : shape = (batch_size, out_channels, height, width)
    """
    assert len(x.shape) == 4, "x not in 4D"
    assert len(y.shape) == 4, "y not in 4D"
    y=np.flip(y,axis = (-2,-1))
    return fast_convulate(x,y)


def fast_hadamard_weight_calc(x,y):
    """
    x: shape = (batch_size, in_channels, padded_height, padded_weight)
    y: shape = (batch_size, out_channels, modified_del_z_height, modified_del_z_weight)
    out: shape = (out_channels, in_channels, padded_height - modified_del_z_height +1 , padded_weight - modified_del_z_weight+1)
    """
    assert len(x.shape) == 4, "x not in 4D"
    assert len(y.shape) == 4, "y not in 4D"
    assert x.shape[0] == y.shape[0], "batch_size dont match"
    
    n1,m1 = x.shape[-2:]
    n2,m2 = y.shape[-2:]
    n = n1+n2-1
    m = m1+m2-1
    y=np.flip(y, axis = (-2,-1))
    # print("n n1 n2 m m1 m2",n,n1,n2,m,m1,m2)
    x = np.pad(x,((0,0),(0,0),(0,n-n1),(0,m-m1)), mode = 'constant', constant_values = 0)
    y = np.pad(y,((0,0),(0,0),(0,n-n2),(0,m-m2)), mode = 'constant', constant_values = 0)
    
    fx = fft2(x, axes = (-2,-1))
    fy = fft2(y, axes = (-2,-1))
    assert fx.shape [-2:] == (n,m) , "fx shape dont match"
    assert fy.shape [-2:] == (n,m) , "fy shape dont match"
    
    fz = np.einsum("ijkl,imkl->mjkl",fx,fy)
    assert fz.shape == (y.shape[1],x.shape[1],n,m), "fz shape dont match"
    z = np.real(ifft2(fz,axes = (-2,-1)))
    assert z.shape == (y.shape[1],x.shape[1],n,m), "z shape dont match"
    z = z[:,:,n2-1:n1,m2-1:m1]
    
    assert z.shape == (y.shape[1],x.shape[1],n1-n2+1,m1-m2+1), "z shape dont match"
    return z
    

def fast_convulate_x_calc(x,y):
    """
    in x shape: (batch, out_channel, height (appropiate ), width (appropiate ))
    in y shape: (out_channels, in_channels, kernel_height, kernel_width)
    out shape: (batch,in_channels, padded_in_height, padded_in_width)
    """
    assert len(x.shape) == 4, "x not in 4D"
    assert len(y.shape) == 4, "y not in 4D"
    assert x.shape[1] == y.shape[0], "out_channels dont match"
    
    n1,m1 = x.shape[-2:]
    n2,m2 = y.shape[-2:]
    
    n = n1+n2-1
    m = m1+m2-1
    
    # print("n n1 n2 m m1 m2",n,n1,n2,m,m1,m2)
    x = np.pad(x,((0,0),(0,0),(0,n-n1),(0,m-m1)), mode = 'constant', constant_values = 0)
    y = np.pad(y,((0,0),(0,0),(0,n-n2),(0,m-m2)), mode = 'constant', constant_values = 0)
    
    fx = fft2(x, axes = (-2,-1))
    fy = fft2(y, axes = (-2,-1))
    assert fx.shape [-2:] == (n,m) , "fx shape dont match"
    assert fy.shape [-2:] == (n,m) , "fy shape dont match"
    
    fz = np.einsum("ijkl,jmkl->imkl",fx,fy)
    assert fz.shape == (x.shape[0],y.shape[1],n,m), "fz shape dont match"
    z = np.real(ifft2(fz,axes = (-2,-1)))
    assert z.shape == (x.shape[0],y.shape[1],n,m), "z shape dont match"
    z = z[:,:,n2-1:n1,m2-1:m1]
    
    assert z.shape == (x.shape[0],y.shape[1],n1-n2+1,m1-m2+1), "z shape dont match"
    return z


# RELU

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
    


# CONV2D

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
        



class FlatteningLayer():
    
    def __init__(self) -> None:
        pass
    
    def forward(self,x):
        """
			in x: (batch_size, ...)
			out x: (batch_size, ...)
        """
        assert len(x.shape) >= 2, "input shape is not at least 2D"
        self.x_shape = x.shape
        return x.reshape(x.shape[0],-1)
    
    def backward(self,del_z,lr):
        """
            in del_z: (batch_size, ...)
        """
        return del_z.reshape(self.x_shape)

    def clean(self):
        pass



class LinearLayer():
    
    def __init__(self, out_features):
        
        self.out_features = out_features
        self.weights = None
        self.biases = None
        
        # local vars
        self.x = None
        
    def forward(self,x):
        """
            in x: (batch_size, in_features)
            out x: (batch_size, out_features)
        """
        assert len(x.shape) == 2, "input shape is not 2D"
        in_features = x.shape[1]
        batch_size = x.shape[0]
        
        self.x = x
        
        if self.weights is None:    
            # https://cs231n.github.io/neural-networks-2/#init
            # weights shape = (in_features, out_features)
            self.weights = np.random.randn(in_features,self.out_features)*np.sqrt(1.0/in_features)
            # biases shape = (out_features)
            self.biases = np.random.randn(self.out_features)*np.sqrt(1.0/in_features)
        
        assert self.weights.shape == (in_features,self.out_features), "weight shape dont match"
        
        out_x = np.matmul(x,self.weights) + self.biases[np.newaxis,:]

        assert out_x.shape == (batch_size,self.out_features), "output shape dont match"
        return out_x

    def backward(self, del_z, lr):
        """
            in del_z: (batch_size, out_features)
            lr : learning rate
        """
        assert len(del_z.shape) == 2, "input shape is not 2D"
        assert del_z.shape[1] == self.out_features, "del_z shape dont match"
        # https://www.adityaagrawal.net/blog/deep_learning/bprop_fc
        batch_size = del_z.shape[0]
        
        del_w = np.matmul(self.x.T, del_z)/batch_size
        assert del_w.shape == self.weights.shape, "del_w shape dont match"
        del_b = np.sum(del_z, axis = 0)/batch_size
        assert del_b.shape == self.biases.shape, "del_b shape dont match"
        del_x = np.matmul(del_z, self.weights.T)
        assert del_x.shape == self.x.shape, "del_x shape dont match"
        
        self.weights -= lr*del_w
        self.biases -= lr*del_b
        
        return del_x
        
    def clean(self):
        self.x = None
    


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
        
        # local variables
        self.x = None
        self.out_x = None
        
    def forward(self,x):
        """
            input shape = (batch_size, in_channels, height, width)
            output shape = (batch_size, in_channels, out_height, out_width)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        self.x_shape = x.shape
        self.x = x
        out_shape =( (x.shape[2] - self.kernel_shape[0])//self.stride[0] + 1,\
                     (x.shape[3] - self.kernel_shape[1])//self.stride[1] + 1)
        
        strided_x = as_strided(x, 
            strides=(x.strides[0], x.strides[1] , x.strides[2] * self.stride[0], 
                     x.strides[3] * self.stride[1] , x.strides[2] , x.strides[3] ),
            shape = (x.shape[0], x.shape[1], out_shape[0], out_shape[1], self.kernel_shape[0], self.kernel_shape[1])
                        )
        out_x = strided_x.max(axis=(4,5))
        self.out_x = out_x
        assert out_x.shape == (x.shape[0], x.shape[1], out_shape[0], out_shape[1])
        return out_x

    def backward(self, del_z, lr):
        """
        in del_z: (batch_size, in_channels, out_height, out_width)
        out del_x: (batch_size, in_channls, height, width)
        """
        assert len(del_z.shape) == 4 , "del_z is not 4D"
        del_x = np.zeros(self.x_shape)
        
        for i in range(self.kernel_shape[0]):
            for j in range(self.kernel_shape[1]):
                # print("shape of x:",self.x.shape)
                # print("shape of out_x, ",self.out_x.shape)
                # print("self stride ",self.stride)
                # print("self.out_x ",self.out_x.shape)
                ins = self.x[:,:,i:i+self.out_x.shape[2]*self.stride[0]:self.stride[0],j:j+self.out_x.shape[3]*self.stride[1]:self.stride[1]]
                # print("i=",i,"j=",j)
                # print("shape of ins:",ins.shape)
                # print("self out x: ",self.out_x.shape)
                multiplier = np.allclose(ins,self.out_x)
                
                del_x[:,:,i:i+self.out_x.shape[2]*self.stride[0]:self.stride[0],j:j+self.out_x.shape[3]*self.stride[1]:self.stride[1]] += multiplier * del_z
        
        return del_x
        
    def clean(self):
        self.x = None
        self.out_x = None


class SoftMax():
    
    def forward(self,x):
        """
            in x: (samples, classes)
               out x: (samples, classes)
        """
        assert len(x.shape) == 2, "The input of SoftMax must be 2-D"
        mx = np.max(x, axis = 1, keepdims = True)
        exp = np.exp(x-mx)
        # print("SoftMax exp : ",exp)
        return exp / np.sum( exp , axis = 1, keepdims = True)
    
    def backward(self,y_pred_minus_y_true, lr):
        """
            in y_pred: (samples, classes)
            out y_pred: (samples, classes)
        """
        assert len(y_pred_minus_y_true.shape) == 2, "The input of SoftMax must be 2-D"
        return y_pred_minus_y_true

    def clean(self):
        pass



class CNN():
    
    def __init__(self) -> None:
        self.layers = []
        self.log_loss = None
        self.name = 'CNN'
    
    def add_layer(self,layer):
        self.layers.append(layer)
        
    # def cross_entrpy_loss(self,y_pred,y_true):
    #     # print("y_pred:",y_pred)
    #     return np.sum(-1 * np.sum(y_true * np.log(y_pred), axis=0))
    
    
    
    def train(self,x,y_true,lr):
        """
            in: x shape = (batch_size, channels, height, width)
            in: y_true shape = (batch_size, classes)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        assert len(y_true.shape) == 2, "y_true shape is not 2D"
        for layer in self.layers:
            x = layer.forward(x)
            # print("train output shape: ",x.shape)
        y = x
        # print("shape of y_true",y_true.shape)
        # print("train output:",y)
        # print(f"Cross entropy loss: {self.cross_entrpy_loss(y,y_true)}")
        
        del_z = y - y_true
        # print("shape of del_z",del_z.shape)
        for layer in reversed(self.layers):
            del_z = layer.backward(del_z,lr)
            # print("train del_z shape: ",del_z.shape)
    
    def predict(self,x):
        """
            in: x shape = (batch_size, channels, height, width)
            in: y_true shape = (batch_size, classes)
        """
        assert len(x.shape) == 4, "input shape is not 4D"
        for layer in self.layers:
            x= layer.forward(x)
        return x

    def clean(self):
        for layer in self.layers:
            layer.clean()



def get_lenet():
    m = CNN()
    m.add_layer(Conv2d(out_channels=6,kernel_size=(5,5), stride=1,padding=2))
    m.add_layer(ReLU())
    m.add_layer(MaxPool2d(kernel_size=(2,2), stride=2))
    m.add_layer(Conv2d(out_channels=16,kernel_size=(5,5), stride=1,padding=2))
    m.add_layer(ReLU())
    m.add_layer(MaxPool2d(kernel_size=(2,2), stride=2))
    m.add_layer(FlatteningLayer())
    m.add_layer(LinearLayer(out_features=120))
    m.add_layer(ReLU())
    m.add_layer(LinearLayer(out_features=84))
    m.add_layer(ReLU())
    m.add_layer(LinearLayer(out_features=10))
    m.add_layer(SoftMax())
    m.name = 'LeNet'    
    return m


def get_shnet():
    m = CNN()
    m.add_layer(Conv2d(out_channels=32,kernel_size=(3,3), stride=(1,1),padding=0))
    m.add_layer(ReLU())
    m.add_layer(MaxPool2d(kernel_size=(2,2), stride=(2,2)))
    m.add_layer(FlatteningLayer())
    m.add_layer(LinearLayer(out_features=100))
    m.add_layer(ReLU())
    m.add_layer(LinearLayer(out_features=10))
    m.add_layer(SoftMax())
    m.name = 'ShNet'
    return m




import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def crop_image(img):
    # mn = img.min()
    # mx = img.max()
    # print("mn  ",mn)
    # print("mx  ",mx)
    threshold = 122
    img = img * (img > threshold)
    tmp = np.where(img > 0)
    if len(tmp[0])==0 or len(tmp[1])==0:
        return img
    
    min_row = tmp[0].min()
    max_row = tmp[0].max()
    min_col = tmp[1].min()
    max_col = tmp[1].max()
    # print("min_row  ",min_row)
    # print("max_row  ",max_row)
    # print("min_col  ",min_col)
    # print("max_col  ",max_col)
    
    # crop image to bounding box [min_row, max_row, min_col, max_col]
    if min_row<=max_row and min_col<=max_col:
        img = img[min_row:max_row+1, min_col:max_col+1]
    return img



def get_image(image_shape,image_file_path):
        
    if os.path.exists(image_file_path):
        # print("File exists")
        img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        img = 255-img
        
        img = crop_image(img)
        # print(img)
        # plt.imshow(img, cmap='gray', interpolation='bicubic')
        # plt.show()
        img = cv2.resize(img, image_shape)
        # cv2.dilate(img, np.ones((2,2),np.uint8), iterations = 1)
        img = np.expand_dims(img, axis=0)
        
        img=img.astype(np.float32)
        
        img /= 255
        return img
        
    else:
        print("File does not exist:",image_file_path)
        return None


import tqdm
import sys
import os
import pickle as pk
import pandas as pd
if __name__  == '__main__':
    # parse folder path from command line
    if len(sys.argv) < 2:
        print("Usage: python3 test_1705003.py <folder_path>")
        exit(1)
    
    base_path = sys.argv[1]
    # base_path = './../resource/NumtaDB_with_aug/training-d'
    print("base_path: ",base_path)
    
    # load model
    m = pk.load(open("1705003_model.pickle","rb"))
    print("model loaded")
    results = pd.DataFrame(columns = ["FileName","Digit"])
    
    for files in tqdm.tqdm(os.listdir(base_path),"processing files"):
        _,ext = os.path.splitext(files)
        # print("ext: ",ext)
        if ext.endswith('png') or ext.endswith('jpg'):
            file_path = os.path.join(base_path,files)
            # print("file_path: ",file_path)
            file_name = os.path.basename(file_path)
            x = get_image((8,8),file_path)
            if x is None:
                continue
            x = x[np.newaxis,:]
            y_onehot = m.predict(x)
            y = np.argmax(y_onehot)
            # print("file_name: ",file_name)
            # print("y: ",y)
            # plt.imshow(x[0,0,:,:], cmap='gray', interpolation='bicubic')
            # plt.show()
            results = pd.concat([results,pd.DataFrame({"FileName":file_name,"Digit":y},index=[0])],ignore_index=True)
    results.to_csv("1705003_prediction.csv",index=False)