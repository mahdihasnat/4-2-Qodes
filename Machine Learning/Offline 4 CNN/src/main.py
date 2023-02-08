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

import pickle as pk
import pandas as pd

from sklearn import model_selection as skms

from cnn import get_lenet


if __name__ == '__main__':
    np.random.seed(0)
    
    m = get_lenet()
    
    x,y = load_dataset(image_shape=(32,32),sample_bound=100)
    epoch = 5
    batch_size = 128
    total_sample = x.shape[0]
    lr = 0.01
    train_ratio = 0.8
    
    # shuffle split train and validation
    print("shape of x:",x.shape)
    print("shape of y:",y.shape)
    x_train,x_validation,y_train,y_validation = skms.train_test_split(x,y,train_size=train_ratio,shuffle=True,stratify=y)
    
    
    print("Train size: {}".format(x_train.shape[0]))
    print("Validation size: {}".format(x_validation.shape[0]))
    
    total_batch_train = (x_train.shape[0]+batch_size-1)//batch_size
    total_batch_validation = (x_validation.shape[0]+batch_size-1)//batch_size
    
    y_loss_validation=[]
    y_f1=[]
    y_accuracy=[]
    y_loss_train=[]
    stat_d = pd.DataFrame(columns=["epoch","loss","f1","accuracy","loss_train"])
    for i in tqdm.tqdm(range(epoch)):
        print(f"Epoch {i+1}:")
        for j in tqdm.tqdm(range(total_batch_train), "training"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_train.shape[0])
            m.train(x_train[start:end],y_train[start:end],lr)
        
        y_pred = np.zeros(y_validation.shape)
        
        for j in tqdm.tqdm(range(total_batch_validation), "predicting validation"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_validation.shape[0])
            y_pred[start:end]=m.predict(x_validation[start:end])
        
        
        
        loss_value = skm.log_loss(y_true = y_validation,y_pred = y_pred)
        y_loss_validation.append(loss_value)
        
        y_pred_value = np.argmax(y_pred,axis=1)
        y_true_value = np.argmax(y_validation,axis=1)
        
        accuracy_value = skm.accuracy_score(y_true = y_true_value,y_pred = y_pred_value)
        y_accuracy.append(accuracy_value)
        
        f1_score_value = skm.f1_score(y_true = y_true_value,y_pred = y_pred_value,average='macro')
        y_f1.append(f1_score_value)
        
        confusion_matrix = skm.confusion_matrix(y_true = y_true_value,y_pred = y_pred_value)
        
        # train predict
        y_pred = np.zeros(y_train.shape)
        for j in tqdm.tqdm(range(total_batch_train),"predicting train"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_train.shape[0])
            y_pred[start:end]=m.predict(x_train[start:end])
        
        loss_value_train = skm.log_loss(y_true = y_train,y_pred = y_pred)
        y_loss_train.append(loss_value_train)
        print("")
        print(f"Train Loss: {loss_value_train}")
        print(f"Validation Loss: {loss_value}")
        print(f"Accuracy: {accuracy_value}")
        print(f"F1 score: {f1_score_value}")
        print(f"Confusion matrix: {confusion_matrix}")
        
        stat_d = pd.concat([stat_d,pd.DataFrame({"epoch":i+1,"loss":loss_value,"f1":f1_score_value,
                                                 "accuracy":accuracy_value,"loss_train":loss_value_train},
                                                index=[0])],ignore_index=True)
        stat_d.to_csv("data/stat_lr_{:.2f}_train_{}_m_{}.csv".format(lr,x_train.shape[0],m.name),index=False)
        
        
        m.clean()
        pk.dump(m,open("models/model_e{}_f1_{:.2f}_acc_{:3f}_m_{}.pkl".format(i+1,f1_score_value,accuracy_value,m.name),"wb"))
        
        
    y_loss_validation = np.array(y_loss_validation)
    y_accuracy = np.array(y_accuracy)
    y_f1 = np.array(y_f1)
    x_axis = np.arange(epoch)
    
    str_pre = "lr_{:.2f}_train_{}_m_{}_e_{}_".format(lr,x_train.shape[0],m.name,epoch)
    
    plt.figure()
    plt.plot(x_axis,y_loss_validation)
    plt.xticks(x_axis)
    plt.xlabel("Epoch")
    plt.title("Validation Cross entropy Loss")
    plt.savefig("figs/"+str_pre+"validation_loss.png")
    
    plt.figure()
    plt.plot(x_axis,y_loss_train)
    plt.xticks(x_axis)
    plt.xlabel("Epoch")
    plt.title("Train Cross entropy Loss")
    plt.savefig("figs/"+str_pre+"train_loss.png")
    
    plt.figure()
    plt.plot(x_axis,y_accuracy)
    plt.xticks(x_axis)
    plt.xlabel("Epoch")
    plt.title("Accuracy")
    plt.savefig("figs/"+str_pre+"validation_acc.png")
    
    plt.figure()
    plt.plot(x_axis,y_f1)
    plt.xticks(x_axis)
    plt.xlabel("Epoch")
    plt.title("F1 score")
    plt.savefig("figs/"+str_pre+"validation_f1.png")
    