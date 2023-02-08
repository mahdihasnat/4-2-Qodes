from cnn import CNN
from convlayer import Conv2d
from activationlayer import ReLU

from maxpoollayer import MaxPool2d

from flatteninglayer import FlatteningLayer

from linearlayer import LinearLayer
from softmaxlayer import SoftMax

from data_handler import load_dataset
from matplotlib import pyplot as plt
import tqdm
from sklearn import metrics as skm

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

def get_lenet():
    m = CNN()
    m.add_layer(Conv2d(out_channels=6,kernel_size=(5,5), stride=1,padding=0))
    m.add_layer(ReLU())
    m.add_layer(MaxPool2d(kernel_size=(2,2), stride=2))
    m.add_layer(Conv2d(out_channels=16,kernel_size=(5,5), stride=1,padding=0))
    m.add_layer(ReLU())
    m.add_layer(MaxPool2d(kernel_size=(2,2), stride=2))
    m.add_layer(FlatteningLayer())
    m.add_layer(LinearLayer(out_features=120))
    m.add_layer(ReLU())
    # m.add_layer(FlatteningLayer())
    m.add_layer(LinearLayer(out_features=84))
    m.add_layer(ReLU())
    # m.add_layer(FlatteningLayer())
    m.add_layer(LinearLayer(out_features=10))
    m.add_layer(SoftMax())
    
    return m

if __name__ == '__main__':
    # np.random.seed(0)
    
    m = get_lenet()
    
    x,y = load_dataset(image_shape=(32,32),sample_bound=10000)
    epoch = 20
    batch_size = 64
    total_sample = x.shape[0]
    lr = 0.01
    train_ratio = 0.8
    
    # shuffle split train and validation
    perm = np.random.permutation(total_sample)
    x = x[perm]
    y = y[perm]
    
    train_size = int(total_sample*train_ratio)
    x_train = x[:train_size]
    x_validation = x[train_size:]
    y_train = y[:train_size]
    y_validation = y[train_size:]
    del x
    del y
    
    total_batch_train = (x_train.shape[0]+batch_size-1)//batch_size
    total_batch_validation = (x_validation.shape[0]+batch_size-1)//batch_size
    
    y_loss=[]
    y_f1=[]
    y_accuracy=[]
    for i in tqdm.tqdm(range(epoch)):
        print(f"Epoch {i+1}:")
        for j in tqdm.tqdm(range(total_batch_train), "training"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_train.shape[0])
            m.train(x_train[start:end],y_train[start:end],lr)
            
        y_pred = np.zeros(y_validation.shape)
        for j in tqdm.tqdm(range(total_batch_validation), "predicting"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_validation.shape[0])
            y_pred[start:end]=m.predict(x_validation[start:end])
        
        loss_value = skm.log_loss(y_true = y_validation,y_pred = y_pred)
        y_loss.append(loss_value)
        
        
        y_pred_value = np.argmax(y_pred,axis=1)
        y_true_value = np.argmax(y_validation,axis=1)
        
        accuracy_value = skm.accuracy_score(y_true = y_true_value,y_pred = y_pred_value)
        y_accuracy.append(accuracy_value)
        
        f1_score_value = skm.f1_score(y_true = y_true_value,y_pred = y_pred_value,average='macro')
        y_f1.append(f1_score_value)
        
        print(f"Loss: {loss_value}")
        print(f"Accuracy: {accuracy_value}")
        print(f"F1 score: {f1_score_value}")
        
        
    y_loss = np.array(y_loss)
    y_accuracy = np.array(y_accuracy)
    y_f1 = np.array(y_f1)
    x_axis = np.arange(epoch)
    
    plt.figure()
    plt.plot(x_axis,y_loss)
    plt.xlabel("Epoch")
    plt.title("Cross entropy Loss")
    
    plt.figure()
    plt.plot(x_axis,y_accuracy)
    plt.xlabel("Epoch")
    plt.title("Accuracy")
    
    plt.figure()
    plt.plot(x_axis,y_f1)
    plt.xlabel("Epoch")
    plt.title("F1 score")
    
    
    plt.show()
    
    