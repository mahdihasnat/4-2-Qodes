from gmm2 import GMM
from data_handler import load_dataset
X =load_dataset()
g = GMM(k=5,max_iter = 1000)
g.fit(X)