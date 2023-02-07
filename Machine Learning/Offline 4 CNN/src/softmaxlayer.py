import numpy as np

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

if __name__ == '__main__':
    
    x_shape = (100,10)
    x = np.random.randint(-100,100,x_shape)
    print(x.shape)
    s = SoftMax()
    z = s.forward(x)
    print(z.shape)
    