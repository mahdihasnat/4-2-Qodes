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
        theta_trans_x = np.dot(x, self.theta)
        return 1 / (1 + np.exp(-theta_trans_x))
    
    def __fit_with_cost_minimization(self, X, y):
        
        for i in range(self.max_iter):
            h = self.__hypothesis(X)
            h_minus_y = h - y
            grad = np.dot(X.T, h_minus_y) / X.shape[0]
            self.theta -= grad* self.alpha
            if i%1000 == 0:
                print("iteration ", i, "cost ", np.sum(-y*np.log(h) - (1-y)*np.log(1-h)))
    
    def __fit_with_likelihood_maximization(self, X, y):
        
        for i in range(self.max_iter):
            h = self.__hypothesis(X)
            y_minus_h = y - h
            grad = np.dot(X.T, y_minus_h) / X.shape[0]
            self.theta += grad* self.alpha
            if i%1000 == 0:
                z = np.dot(X, self.theta)
                print("iteration ", i, "log likelihood ", np.sum(y*z - np.log(1 + np.exp(z))))
    
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
        
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # self.theta = np.random.rand(X.shape[1],1)
        self.theta = np.zeros((X.shape[1],1))
        
        self.__fit_with_cost_minimization(X, y)
        self.__fit_with_likelihood_maximization(X, y)
        
        
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