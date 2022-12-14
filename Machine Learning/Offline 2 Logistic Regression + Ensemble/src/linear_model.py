import numpy as np
class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        assert 'alpha' in params , "alpha not found"
        assert 'max_iter' in params , "max_iter not found"
        
        self.alpha = params['alpha']
        self.max_iter = params['max_iter']
        
        
    def __hypothesis(self, x):
        """
        :param x: input vector
        :param theta: weights
        :return: sigmoid of x
        """
        # print("shape of x ", x.shape)
        # print("shape of theta ", self.theta.shape)
        theta_trans_x = np.dot(x, self.theta)
        # print("shape of theta_trans_x ", theta_trans_x.shape)
        return 1 / (1 + np.exp(-theta_trans_x))
    
    
    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0], "X and y should have same number of rows"
        assert len(X.shape) == 2 , "X should be 2D"
        assert len(y.shape) == 2 , "y should be 2D"
        assert y.shape[1] == 1 , "y should be n x 1"
        
        # y.reshape((y.shape[0],))
        # print("shape of Y ", y.shape)
        # todo: implement
        # add bias
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # initialize random weights
        self.theta = np.random.rand(X.shape[1],1)
        # print("shape of theta ", self.theta.shape)
        # print("theta: ", self.theta)
        
        for i in range(self.max_iter):
            h = self.__hypothesis(X)
            # print("shape of h ", h.shape)
            # print("shape of y ", y.shape)
            h_minus_y = h - y
            # print("h: ", h)
            # print("y: ", y)
            # print("h_minus_y: ", h_minus_y)
            # print("shape of h_minus_y ", h_minus_y.shape)
            grad = np.dot(X.T, h_minus_y) / X.shape[0]
            # print("shape of grad ", grad.shape)
            # print("grad:",grad)
            # print("shape of alpha ", self.alpha.shape)
            # gun = grad * self.alpha
            # print("shape of gun ", gun.shape)
            # # print("gun: ", gun)
            # print("shape of theta ", self.theta.shape)
            self.theta -= grad* self.alpha
            # print("shape of theta ", self.theta.shape)
        
    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        # add bias
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # print("shape of X ", X.shape)
        h = self.__hypothesis(X)
        # print("shape of h ", h.shape)
        return np.where(h >= 0.5, 1, 0)