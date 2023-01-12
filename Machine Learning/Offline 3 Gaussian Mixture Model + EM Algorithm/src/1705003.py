import numpy as np
from data_handler import load_dataset
from mixture_model import GMM

def main():
    x = load_dataset()
    
    print("type x = ", type(x))
    print("shape x = ", x.shape)
    
    params = {}
    params['n_components'] = 3
    params['max_iter'] = 100
    params['tol'] = 1e-6
    g = GMM(**params)
    g.fit(x)
    y = g.predict(x)
    print(y)

if __name__ == '__main__':
    main()
