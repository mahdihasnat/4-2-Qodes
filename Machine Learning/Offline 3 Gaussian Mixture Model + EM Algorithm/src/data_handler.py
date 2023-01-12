import numpy as np
def load_dataset():
    filename = 'data6D.txt'
    data = np.genfromtxt(filename, delimiter=' ', skip_header=0,dtype=np.float64)
    # discard after 30 row
    # data = data[:10]
    return data