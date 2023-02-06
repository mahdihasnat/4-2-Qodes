import unittest
import numpy as np

from convlayer import Conv2d

def conv_forward_brute(x,weight,biases,out_channels, kernel_shape, stride, padding):
	assert len(x.shape) == 4, "input shape is not 4D"
	
	# print("x_shape: ",x.shape)
	out_shape =( (x.shape[2] + 2*padding[0] - kernel_shape[0])//stride[0] + 1,\
                     (x.shape[3] + 2*padding[1] - kernel_shape[1])//stride[1] + 1)
	# print("out_shape: ",out_shape)
	
	assert out_shape[0] > 0 and out_shape[1] > 0, "output shape is negative"
	padded_x = np.pad(x , ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])),\
                            mode='constant', constant_values=0)
	out_x = np.zeros((x.shape[0],out_channels,out_shape[0],out_shape[1]))
	for i in range(x.shape[0]):
		for j in range(out_channels):
			for k in range(out_shape[0]):
				for l in range(out_shape[1]):
					out_x[i][j][k][l]=0
					for m in range(x.shape[1]):
						for n in range(kernel_shape[0]):
							for o in range(kernel_shape[1]):
								out_x[i][j][k][l] += padded_x[i][m][k*stride[0]+n][l*stride[1]+o]*weight[j][m][n][o]
					out_x[i][j][k][l] += biases[j]
	return out_x
	

class Conv2dTest(unittest.TestCase):
    
    def test_random_conv(self):
        x_shape = (2,5,128,128)
        k_shape = (12,13)
        out_channels = 5
        stride = (5,7)
        padding = (6,9)
        x = np.random.randint(-100,100,x_shape)
        w = np.random.randint(-100,100,(out_channels,x_shape[1],k_shape[0],k_shape[1]))
        # w = np.zeros((out_channels,x_shape[1],k_shape[0],k_shape[1]))
        b = np.random.randint(-100,100,out_channels)
        # b = np.zeros(out_channels)
        # print("x test:",x)
        # print("w test:",w)
        # print("b test:",b)
        cl = Conv2d(out_channels,k_shape,stride,padding)
        cl.weights = w
        cl.biases = b
        a = cl.forward(x)
        b = conv_forward_brute(x,w,b,out_channels,k_shape,stride,padding)
        print("a: ",a)
        print("b: ",b)
        print("a.shape: ",a.shape)
        print("b.shape: ",b.shape)
        self.assertTrue(np.allclose(a,b), "forward pass is not correct")

if __name__ == '__main__':
	unittest.main()