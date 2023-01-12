import numpy as np

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
                print("val = ", val)
                assert val.shape == ()
                self.r[i][j] = self.pi[j] * val
            self.r[i] /= np.sum(self.r[i])
        
        
    def m_step(self, X):
        assert X.shape == (self.n, self.d)

        Nk = np.sum(self.r, axis=0)
        assert Nk.shape == (self.k,)
        
        self.pi = Nk / self.n
        assert self.pi.shape == (self.k,)
        
    def fit(self, X):
        self.init(X)
        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
    
    def multivariate_normal(self, x, mu, sigma):
        det = np.linalg.det(sigma) 
        assert det.shape == ()
        nom = np.exp(-0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu)))
        assert nom.shape == ()
        denominator = np.sqrt(det *((2 * np.pi) ** self.d))
        assert denominator.shape == ()
        return nom / denominator