import sys

import pickle as pk
from cnn import get_lenet


if __name__ == '__main__':
    
    m = get_lenet()
    m.layers[0].w = 100
    pickle_file = 'models/model_e100_f1_0.00_acc_0.100_lr_1e-06_m_CNN.pkl'
    pk.dump(m,open(pickle_file,'wb'))
    
    m = pk.load(open(pickle_file,'rb'))
    assert m.layers[0].w == 100, "The weight of the first layer is not 100"
    