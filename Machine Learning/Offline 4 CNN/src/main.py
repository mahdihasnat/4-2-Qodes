from cnn import CNN
from convlayer import Conv2d
from activationlayer import ReLU

from maxpoollayer import MaxPool2d

from flatteninglayer import FlatteningLayer

from linearlayer import LinearLayer
from softmaxlayer import SoftMax

from data_handler import load_dataset

import tqdm

import numpy as np
import cv2
# Conv 6 5 1 2
# ReLU
# Pool 2 2
# Conv 12 5 1 0
# ReLU
# Pool 2 2
# Conv 100 5 1 0
# ReLU
# Flatten
# FC 10
# Softmax

if __name__ == '__main__':
    # np.random.seed(0)
    
    m = CNN()
    m.add_layer(Conv2d(out_channels=5,kernel_size=(5,5), stride=1,padding=2))
    m.add_layer(ReLU())
    m.add_layer(MaxPool2d(kernel_size=(2,2), stride=2))
    m.add_layer(Conv2d(out_channels=12,kernel_size=(5,5), stride=1,padding=0))
    m.add_layer(ReLU())
    m.add_layer(MaxPool2d(kernel_size=(2,2), stride=2))
    m.add_layer(Conv2d(out_channels=100,kernel_size=(5,5), stride=1,padding=0))
    m.add_layer(ReLU())
    m.add_layer(FlatteningLayer())
    m.add_layer(LinearLayer(out_features=10))
    m.add_layer(SoftMax())
    
    # # Lanet
    # m.add_layer(Conv2d(out_channels=6,kernel_size=(5,5), stride=1,padding=2))
    # m.add_layer(ReLU())
    # m.add_layer(MaxPool2d(kernel_size=(2,2), stride=2))
    # m.add_layer(Conv2d(out_channels=12,kernel_size=(5,5), stride=1,padding=0))
    
    
    x,y = load_dataset(image_shape=(28,28))
    epoch = 100
    batch_size = 100
    total_sample = x.shape[0]
    total_batch = (total_sample+batch_size-1)//batch_size
    lr = 0.01
    for i in range(epoch):
        print(f"Epoch {i+1}:")
        for j in range(total_batch):
            start = j*batch_size
            end = min((j+1)*batch_size,total_sample)
            m.train(x[start:end],y[start:end],lr)
            # for k in range(start,end):
            #     # show image and prediction
            #     print("y_true:",y[k:k+1])
            #     print("y_pred:",m.predict(x[k:k+1]))
            #     print("y_pred:",np.argmax(m.predict(x[k:k+1])))
            #     # cv2.imshow("image",x[k].reshape(28,28))
            #     # input()
            #     print("")
            

    