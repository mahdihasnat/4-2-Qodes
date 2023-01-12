
class GMM:
	
	def __init__(self,**kwargs) -> None:
		assert 'n_components' in kwargs, "n_components not found"
		assert 'max_iter' in kwargs, "max_iter not found"
		assert 'tol' in kwargs, "tol not found"
		
		self.n_components = kwargs['n_components']
		self.max_iter = kwargs['max_iter']
		self.tol = kwargs['tol']
	
	def fit(self, X):
	 	pass