from gmm2 import GMM
from data_handler import load_dataset
X =load_dataset()
lo = 1 
hi = 10
for k in range(lo,hi+1):
	g = GMM(k=k,max_iter = 1000)
	g.fit(X)