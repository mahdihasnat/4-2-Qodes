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
    
    def draw_gaussian(self, X):
        
        plt.scatter(X[:, 0], X[:, 1], c=self.r.argmax(axis=1), cmap='viridis', s=40, edgecolor='k',
                    alpha=0.2, marker='.')
        x, y = np.mgrid[np.min(X[:, 0]):np.max(X[:, 0]):.01, np.min(X[:, 1]):np.max(X[:, 1]):.01]
        positions = np.dstack((x, y))
        for i in range(self.k):
            
            plt.contour(x, y, self.multivariate_normal(positions, self.mu[i], self.sigma[i])
                        , colors='black', alpha=0.6, linewidths=1)
            plt.show()

    def animate(self, X):
        
        self.init(X)
        assert self.d == 2 , "only support 2D data"

        fig = plt.figure()
        plt.ion()

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

            
            # x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
            # y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
            # x, y = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
            # X_mesh, Y_mesh = np.meshgrid(x, y)
            # print("shape of X_mesh = ", X_mesh.shape)
            # print("shape of Y_mesh = ", Y_mesh.shape)
            # # Calculate the probability density function for each Gaussian component on the meshgrid
            # Z = np.zeros((len(x), len(y)))
            # for i in range(self.k):
            #     Z += self.pi[i] * self.multivariate_normal(np.column_stack((X_mesh.ravel(), Y_mesh.ravel())), 
            #                                                self.mu[i], self.sigma[i])
            #     Z = Z.reshape(X_mesh.shape)

            # # Plot the contour lines
            # plt.contour(X_mesh, Y_mesh, Z)
            # plt.show()
            
            
            
            plt.clf()
            self.draw_gaussian(X)
            plt.title("Iteration {}".format(i))
            plt.pause(0.005)
    

        plt.ioff()

    