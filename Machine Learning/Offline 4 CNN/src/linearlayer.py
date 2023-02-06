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
        
        if self.weights is None:    
            # https://cs231n.github.io/neural-networks-2/#init
            # weights shape = (in_features, out_features)
            self.weights = np.random.randn(in_features,self.out_features)*np.sqrt(1.0/in_features)
            # biases shape = (out_features)
            self.biases = np.random.randn(self.out_features)*np.sqrt(1.0/in_features)
        
        assert self.weights.shape == (in_features,self.out_features), "weight shape dont match"
        
        out_x = np.matmul(x,self.weights) + self.biases
        
        return out_x
    

if __name__ == '__main__':
    x_shape = (100,15)
    out_features = 10
    x = np.random.randint(-100,100,x_shape)
    l = LinearLayer(out_features)
    z = l.forward(x)
    print(z.shape)
        
  
  
  