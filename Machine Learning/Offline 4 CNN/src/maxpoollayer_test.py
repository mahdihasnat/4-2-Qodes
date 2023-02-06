import unittest
import numpy as np

from maxpoollayer import MaxPool2d

def max_pool_brute(x,kernel_shape, stride):
    assert len(x.shape) == 4, "input shape is not 4D"
    assert len(stride) == 2, "stride is not 2D"
    assert len(kernel_shape) == 2, "kernel shape is not 2D"
    
    out_shape =( (x.shape[2] - kernel_shape[0])//stride[0] + 1,\
                     (x.shape[3] - kernel_shape[1])//stride[1] + 1)
    
    out_x = np.zeros((x.shape[0], x.shape[1], out_shape[0], out_shape[1]))
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(out_shape[0]):
                for l in range(out_shape[1]):
                    out_x[i,j,k,l] = np.max(x[i,j,k*stride[0]:k*stride[0]+kernel_shape[0], l*stride[1]:l*stride[1]+kernel_shape[1]])
    
    return out_x

class MaxPoolLayerTest(unittest.TestCase):
    
    def test_random_maxpool(self):
        x_shape = (100,15,32,32)
        k_shape = (3,7)
        stride = (6,5)
        x = np.random.randint(-100,100,x_shape)
        maxpoollayer = MaxPool2d(k_shape,stride)
        a = maxpoollayer.forward(x)
        b = max_pool_brute(x,k_shape,stride)
        self.assertTrue(np.allclose(a,b), "forward pass is not correct")


if __name__ == '__main__':
    unittest.main()