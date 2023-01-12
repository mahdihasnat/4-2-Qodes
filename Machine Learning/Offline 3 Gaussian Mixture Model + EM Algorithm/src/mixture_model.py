import numpy as np

from scipy.stats import multivariate_normal
DTYPE = np.float64
class GMM:
    
    def __init__(self,**kwargs) -> None:
        assert 'n_components' in kwargs, "n_components not found"
        assert 'max_iter' in kwargs, "max_iter not found"
        assert 'tol' in kwargs, "tol not found"
        
        self.n_components = kwargs['n_components']
        self.max_iter = kwargs['max_iter']
        self.tol = kwargs['tol']
    
    def init(self, X):
        self.n, self.d = X.shape
        
        # pi[i] = P(z = i)
        self.pi = np.full(shape=self.n_components, fill_value=1/self.n_components)
        
        # print("pi = ", self.pi)
        print("shape of pi ", self.pi.shape)

        
        # splitting data
        new_X = np.array_split(X, self.n_components)
        # print("new_X = ", new_X)
        
        # mean 
        self.mu = np.array([np.mean(x, axis=0) for x in new_X])
        
        # print("mu = ", self.mu)
        print("shape of mu ", self.mu.shape)
        
        # covariance
        # self.sigma = np.array([np.cov(x.T) for x in new_X])
        self.sigma = np.tile(np.eye(self.d), (self.n_components, 1, 1))
        
        
        print("sigma = ", self.sigma)
        print("shape of sigma ", self.sigma.shape)

        del new_X
        
    def e_step(self, X):
        # gamma[i][j] = P( z = i | x )
        # print("X = ", X)
        self.gamma = np.array([ np.array( [ self.multivariate_normal(X[j],self.mu[i],self.sigma[i]) for j in range(self.n) ])
                for i in range(self.n_components) ])
        
        # print("gamma = ", self.gamma)
        # print("shape of gamma ", self.gamma.shape)
        
        row_sum = np.sum(self.gamma, axis = 0)
        # print("row_sum = ", row_sum)
        # print("shape of row_sum ", row_sum.shape)
        
        self.gamma = self.gamma / row_sum
        # print("gamma = ", self.gamma)
        # print("shape of gamma ", self.gamma.shape)
        
    
    def m_step(self, X):
        column_sum = np.sum(self.gamma, axis = 1)
        # print("column_sum = ", column_sum)
        # print("shape of column_sum ", column_sum.shape)
        
        
        self.mu = np.array([ np.dot(self.gamma[i], X) /column_sum[i] for i in range(self.n_components) ])
        # print("mu = ", self.mu)
        # print("shape of mu ", self.mu.shape)
        
        self.sigma = np.array([ np.cov(X.T, aweights=(self.gamma[i]), ddof=0) for i in range(self.n_components) ])
        # print("sigma = ", self.sigma)
        # print("shape of sigma ", self.sigma.shape)

        self.pi = column_sum / self.n
        # print("pi = ", self.pi)
        # print("shape of pi ", self.pi.shape)
        
    def log_likelihood(self, X):
        ret = 0
        for i in range(self.n):
            now = 0
            for j in range(self.n_components):
                now+=self.pi[j]*self.multivariate_normal(X[i],self.mu[j],self.sigma[j])
            ret+=np.log(now)
        return ret
    
 
    def fit(self, X):
        self.init(X)
        for i in range(self.max_iter):
        # for i in range(1):
            self.e_step(X)
            self.m_step(X)
            ll = self.log_likelihood(X)
            print("Log likelihood = ", ll)
    
    def predict(self, X):
        assert self.mu is not None, "Model not trained"
        assert self.sigma is not None, "Model not trained"
        assert self.pi is not None, "Model not trained"
        assert X.shape[1] == self.d, "Dimension mismatch"
        # return [self.multivariate_normal() for i in range(self.n_components)]
    
    def multivariate_normal(self, x, mu, sigma):
        
        return multivariate_normal.pdf(x, mean=mu, cov=sigma, allow_singular=True)