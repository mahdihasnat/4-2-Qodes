import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class GMM:
    
    def __init__(self,**kwargs):
        
        assert 'k' in kwargs, "k is not specified"
        assert 'max_iter' in kwargs, "max_iter is not specified"
        assert 'tol' in kwargs, "tol is not specified"
        
        
        self.k = kwargs['k']
        self.max_iter = kwargs['max_iter']
        self.tol = kwargs['tol']
        self.verbose = kwargs.get('verbose', False)
        
    def init(self, X):
        self.n, self.d = X.shape
        
        self.pi = np.full(self.k, 1/self.k)
        
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [  X[row_index,:] for row_index in random_row ]
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ]
        
    
    def e_step(self, X):
        # write r in vectorized form
        self.r = np.zeros((self.n, self.k))
        for i in range(self.k):
            
            self.r[:,i] = self.multivariate_normal(X, self.mu[i], self.sigma[i])
            
        
        numerator = self.r * self.pi
        den = np.sum(numerator, axis=1)[:, np.newaxis]
        self.r = numerator / den
        assert self.r.shape == (self.n, self.k)
            
    def m_step(self, X):
        
        for i in range(self.k):
            rk = self.r[:,[i]]
            assert rk.shape == (self.n, 1)
            total_rk = np.sum(rk)
            self.mu[i] = (X*rk).sum(axis = 0) / total_rk
            assert self.mu[i].shape == (self.d,)
            self.sigma[i] = np.cov(X.T, aweights=(rk/total_rk).flatten(), bias=True)
            assert self.sigma[i].shape == (self.d, self.d)
        
        self.pi = self.r.mean(axis=0)
        assert self.pi.shape == (self.k,)
    
    def log_likelihood(self, X):
        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            likelihood[:,i] = self.multivariate_normal(X, self.mu[i], self.sigma[i])
        assert likelihood.shape == (self.n, self.k)
        likelihood = likelihood * self.pi
        assert likelihood.shape == (self.n, self.k)
        return np.sum(np.log(likelihood.sum(axis=1)))
    
    def fit(self, X):
        self.init(X)
        last_log_likelihood = np.inf
        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            ll = self.log_likelihood(X)
            if np.abs(last_log_likelihood-ll) < self.tol:
                break
            last_log_likelihood = ll
            if self.verbose and i % 5 == 0:
                # print("mean = ", self.mu)
                # print("sigma = ", self.sigma)
                print("iter = ", i, "log_likelihood = ", self.log_likelihood(X))
    
    def multivariate_normal(self, x, mu, sigma):
        mvn = multivariate_normal(mean=mu, cov=sigma,allow_singular=True)
        return mvn.pdf(x)


    def predict(self,X):
        pass

    def animate(self, X):
        
        self.init()
        assert self.d == 2 , "only support 2D data"

        fig = plt.figure()
        plt.ion()

        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
            plt.clf()
            plt.title("Iteration {}".format(i))
            plt.pause(0.005)

        plt.ioff()

    