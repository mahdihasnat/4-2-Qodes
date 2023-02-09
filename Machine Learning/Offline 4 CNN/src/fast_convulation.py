import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

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

if __name__ == '__main__':
    batch_size = 15
    in_channels = 5
    grid_shape = (10,20)
    kernel_shape = (7,9)
    out_channels =1
    
    x_shape = (batch_size,in_channels,grid_shape[0],grid_shape[1])
    y_shape = (out_channels,in_channels,kernel_shape[0],kernel_shape[1])
    print("x_shape: ",x_shape)
    print("y_shape: ",y_shape)
    x = np.random.randint(-10,10,x_shape)
    y = np.random.randint(-10,10,y_shape)
    print("x main: ",x)
    print("y main: ",y)
    z = fast_convulate(x,y)
    print("z main: ",z)