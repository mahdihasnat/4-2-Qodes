import numpy as np
from data_handler import load_dataset
from mixture_model import GMM
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np


def test_animate(k):
    X = load_dataset()
    params = {}
    params['k'] = k
    params['max_iter'] = 1000
    params['tol'] = 1e-6
    params['verbose'] = True
    g = GMM(**params)
    g.animate(X)

def main():
    x = load_dataset()
    
    print("type x = ", type(x))
    print("shape x = ", x.shape)
    
    lo = 1 
    hi = 10
    x_vals = np.arange(lo, hi+1, 1)
    y_vals = []
    
    params = {}
    params['k'] = 1
    params['max_iter'] = 1000
    params['tol'] = 1e-2
    params['verbose'] = True
    g = GMM(**params)
    
    for k in range(lo,hi+1):
        g.k = k
        g.fit(x)
        y_vals.append(g.log_likelihood(x))
        # ll = g.log_likelihood(x)
        # print("k = ", k, "ll = ", ll)
        
        # y = g.predict(x)
        # print(y)

    y_vals = np.array(y_vals)
    print("xvals = ", x_vals)
    print("y_vals = ", y_vals)
    plt.plot(x_vals, y_vals)
    plt.xticks(x_vals)
    plt.show()
    
    k_star = int(input("enter k_star: "))
    print("k_star = ", k_star)
    
    g.k = k_star
    g.animate(x)

if __name__ == '__main__':
    test_animate(3)
    # main()
    
