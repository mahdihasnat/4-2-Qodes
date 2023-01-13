import numpy as np
from data_handler import load_dataset
from mixture_model import GMM

import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm

def main():
    x = load_dataset()
    
    print("type x = ", type(x))
    print("shape x = ", x.shape)
    
    for k in range(1,10):
        params = {}
        params['k'] = k
        params['max_iter'] = 20
        params['tol'] = 1e-6
        g = GMM(**params)
        g.fit(x)
        # ll = g.log_likelihood(x)
        # print("k = ", k, "ll = ", ll)
        
        # y = g.predict(x)
        # print(y)

if __name__ == '__main__':
    main()
