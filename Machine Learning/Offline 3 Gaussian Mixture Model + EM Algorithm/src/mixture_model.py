import numpy as np
from scipy.stats import multivariate_normal
class GMM:
    
    def __init__(self,**kwargs):
        
        assert 'k' in kwargs, "k is not specified"
        assert 'max_iter' in kwargs, "max_iter is not specified"
        assert 'tol' in kwargs, "tol is not specified"
        
        self.k = kwargs['k']
        self.max_iter = kwargs['max_iter']
        self.tol = kwargs['tol']
        
    def init(self, X):
        self.n, self.d = X.shape
        
        self.pi = np.full(self.k, 1/self.k)
        assert self.pi.shape == (self.k,)
        
        self.mu = np.random.rand(self.k, self.d)
        
        assert self.mu.shape == (self.k, self.d)
        # initi sigma as identity matrix
        self.sigma = np.array([np.identity(self.d)] * self.k)
        
        assert self.sigma.shape == (self.k, self.d, self.d)
        
    
    def e_step(self, X):
        assert X.shape == (self.n, self.d)
        
        self.r = np.zeros((self.n, self.k))
        for i in range(self.n):
            for j in range(self.k):
                val = self.multivariate_normal(X[i], self.mu[j], self.sigma[j])
                assert val.shape == ()
                self.r[i][j] = self.pi[j] * val
            den = np.sum(self.r[i])
            # if abs(den)<1e-9:
            #     den = 1e-6
            self.r[i] /= den
            
        
        
    def m_step(self, X):
        assert X.shape == (self.n, self.d)

        Nk = np.sum(self.r, axis=0)
        assert Nk.shape == (self.k,)
        
        self.mu = np.zeros((self.k, self.d))
        for i in range(self.k):
            self.mu[i] = np.sum(self.r[:, i].reshape(-1, 1) * X, axis=0) / Nk[i]
        assert self.mu.shape == (self.k, self.d)
        
        self.sigma = np.zeros((self.k, self.d, self.d))
        for i in range(self.k):
            diff = X - self.mu[i]
            assert diff.shape == (self.n, self.d)
            self.sigma[i] = np.dot( (self.r[:,i].reshape(-1,1)*diff).T , diff) / Nk[i]
            assert self.sigma[i].shape == (self.d, self.d)
        
        self.pi = Nk / self.n
        assert self.pi.shape == (self.k,)
    
    def log_likelihood(self, X):
        ret = 0
        for i in range(self.n):
            now = 0
            for j in range(self.k):
                now += self.pi[j] * self.multivariate_normal(X[i], self.mu[j], self.sigma[j])
            ret += np.log(now)
        return ret
    
    
    def fit(self, X):
        self.init(X)
        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            if i % 5 == 0:
                # print("mean = ", self.mu)
                # print("sigma = ", self.sigma)
                print("iter = ", i, "log_likelihood = ", self.log_likelihood(X))
    
    def multivariate_normal(self, x, mu, sigma):
        # det = np.linalg.det(sigma) 
        # assert det.shape == ()
        # if det == 0:
        #     sigma += 0.01 * np.identity(self.d)
        #     det = np.linalg.det(sigma)
        
        # nom = np.exp(-0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu)))
        # assert nom.shape == ()
        # denominator = np.sqrt(det *((2 * np.pi) ** self.d))
        # assert denominator.shape == ()
        # ret = nom / denominator
        # print("ret = ", ret)
        ret2 = multivariate_normal.pdf(x, mean=mu, cov=sigma,allow_singular=True)
        # print("ret2 = ",ret2)
        # assert np.abs(ret - ret2) < 1e-5
        return ret2