import numpy as np

class LinearLayer():
    
    def __init__(self, out_features):
        
        self.out_features = out_features
        self.weights = None
        self.biases = None
        
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
        
        del_w = np.matmul(self.x.T, del_z)
        assert del_w.shape == self.weights.shape, "del_w shape dont match"
        del_b = np.sum(del_z, axis = 0)
        assert del_b.shape == self.biases.shape, "del_b shape dont match"
        del_x = np.matmul(del_z, self.weights.T)
        assert del_x.shape == self.x.shape, "del_x shape dont match"
        self.weights -= lr*del_w
        self.biases -= lr*del_b
        
        return del_x
        
    

if __name__ == '__main__':
    x_shape = (100,15)
    out_features = 10
    x = np.random.randint(-100,100,x_shape)
    l = LinearLayer(out_features)
    z = l.forward(x)
    print(z.shape)
        
  
  
  