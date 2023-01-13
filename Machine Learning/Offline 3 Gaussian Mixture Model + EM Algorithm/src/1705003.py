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
    
    lo = 1 
    hi = 5
    x_vals = np.arange(lo, hi+1, 1)
    y_vals = []
    print("xvals = ", x_vals)
    
    for k in range(lo,hi+1):
        params = {}
        params['k'] = k
        params['max_iter'] = 5
        params['tol'] = 1e-6
        g = GMM(**params)
        g.fit(x)
        y_vals.append(g.log_likelihood(x))
        # ll = g.log_likelihood(x)
        # print("k = ", k, "ll = ", ll)
        
        # y = g.predict(x)
        # print(y)

    y_vals = np.array(y_vals)
    
    plt.plot(x_vals, y_vals)
    plt.show()
    
if __name__ == '__main__':
    main()
