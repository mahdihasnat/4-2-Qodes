import numpy as np
def load_dataset():
    filename = 'data2D.txt'
    data = np.genfromtxt(filename, delimiter=' ', skip_header=0,dtype=np.float64)
    return data