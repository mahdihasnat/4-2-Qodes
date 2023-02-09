from cnn import CNN
from convlayer import Conv2d
from activationlayer import ReLU
from maxpoollayer import MaxPool2d
from flatteninglayer import FlatteningLayer
from linearlayer import LinearLayer
from softmaxlayer import SoftMax
from data_handler import load_dataset, load_test_dataset
from matplotlib import pyplot as plt
import tqdm
from sklearn import metrics as skm
import numpy as np
import cv2
from cnn import get_shnet
import pickle as pk
import pandas as pd

from sklearn import model_selection as skms

from cnn import get_lenet

import seaborn as sns
image_shape = (8,8)
x,y_onehot = load_dataset(image_shape=image_shape,sample_bound=-1)
x_test, y_test_onehot = load_test_dataset(image_shape=image_shape,sample_bound=-1)


def train(lr,epoch):
    
    # use 28x28 for lenet
    # m = get_lenet()
    
    # use 8x8 for shnet
    m = get_shnet()
    
    batch_size = 32
    total_sample = x.shape[0]
    train_ratio = 0.8
    
    # shuffle split train and validation
    print("shape of x:",x.shape)
    print("shape of y:",y_onehot.shape)
    x_train,x_validation,y_train_onehot,y_validation_onehot = skms.train_test_split(x,y_onehot,train_size=train_ratio,shuffle=True,stratify=y_onehot)
    
    y_train = np.argmax(y_train_onehot,axis=1)
    y_validation = np.argmax(y_validation_onehot,axis=1)
    y_test = np.argmax(y_test_onehot,axis=1)
    
    print("Train size: {}".format(x_train.shape[0]))
    print("Validation size: {}".format(x_validation.shape[0]))
    print("Test size: {}".format(x_test.shape[0]))
    str_pre = "lr_{:.7f}_train_{}_m_{}_e_{}_".format(lr,x_train.shape[0],m.name,epoch)
    
    total_batch_train = (x_train.shape[0]+batch_size-1)//batch_size
    total_batch_validation = (x_validation.shape[0]+batch_size-1)//batch_size
    total_batch_test = (x_test.shape[0]+batch_size-1)//batch_size
    
    y_loss_validation=[]
    y_f1=[]
    y_accuracy=[]
    y_loss_train=[]
    y_accuracy_test=[]
    y_f1_test=[]
    stat_d = pd.DataFrame(columns=["epoch","loss_validation","loss_train","f1_validation","accuracy_validation","f1_test","accuracy_test"])
    for i in tqdm.tqdm(range(epoch)):
        print(f"Epoch {i+1}:")
        for j in tqdm.tqdm(range(total_batch_train), "training"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_train_onehot.shape[0])
            m.train(x_train[start:end],y_train_onehot[start:end],lr)
        
        y_pred_onehot = np.zeros(y_validation_onehot.shape)
        
        for j in tqdm.tqdm(range(total_batch_validation), "predicting validation"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_validation_onehot.shape[0])
            y_pred_onehot[start:end]=m.predict(x_validation[start:end])
        
        
        loss_value = skm.log_loss(y_true = y_validation_onehot,y_pred = y_pred_onehot)
        y_loss_validation.append(loss_value)
        
        y_pred = np.argmax(y_pred_onehot,axis=1)
        
        accuracy_value = skm.accuracy_score(y_true = y_validation,y_pred = y_pred)
        y_accuracy.append(accuracy_value)
        
        f1_score_value = skm.f1_score(y_true = y_validation,y_pred = y_pred,average='macro')
        y_f1.append(f1_score_value)
        
        
        confusion_matrix = skm.confusion_matrix(y_true = y_validation,y_pred = y_pred)
        
        # train predict
        y_pred_onehot = np.zeros(y_train_onehot.shape)
        for j in tqdm.tqdm(range(total_batch_train),"predicting train"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_pred_onehot.shape[0])
            y_pred_onehot[start:end]=m.predict(x_train[start:end])
        
        loss_value_train = skm.log_loss(y_true = y_train_onehot,y_pred = y_pred_onehot)
        y_loss_train.append(loss_value_train)
        
        
        # test predict
        y_pred_onehot = np.zeros(y_test_onehot.shape)
        for j in tqdm.tqdm(range(total_batch_test),"predicting test"):
            start = j*batch_size
            end = min((j+1)*batch_size,y_pred_onehot.shape[0])
            y_pred_onehot[start:end]=m.predict(x_test[start:end])
            
        y_pred = np.argmax(y_pred_onehot,axis=1)
        accuracy_value_test = skm.accuracy_score(y_true = y_test,y_pred = y_pred)
        f1_score_value_test = skm.f1_score(y_true = y_test,y_pred = y_pred,average='macro')
        
        
        print("")
        print(f"Train Loss: {loss_value_train}")
        print(f"Validation Loss: {loss_value}")
        print(f"Validation Accuracy: {accuracy_value}")
        print(f"Validation F1 score: {f1_score_value}")
        print(f"Test Accuracy: {accuracy_value_test}")
        print(f"Test F1 score: {f1_score_value_test}")
        print(f"Confusion matrix: {confusion_matrix}")
        
        plt.figure(figsize=(10,10))
        ax = sns.heatmap(confusion_matrix, annot=True)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        plt.savefig("figs/"+str_pre+"i_{}_confusion_matrix.png".format(i+1))
        
        
        stat_d = pd.concat([stat_d,pd.DataFrame({"epoch":i+1,"loss_validation":loss_value,
                                                    "loss_train":loss_value_train,
                                                    "f1_validation":f1_score_value,
                                                    "accuracy_validation":accuracy_value,
                                                    "f1_test":f1_score_value_test,
                                                    "accuracy_test":accuracy_value_test
                                                 },
                                                index=[0])],ignore_index=True)
        stat_d.to_csv("data/stat_lr_{:.6f}_train_{}_m_{}.csv".format(lr,x_train.shape[0],m.name),index=False)
        
        
        m.clean()
        pk.dump(m,open("models/model_e{}_f1_{:.2f}_acc_{:.3f}_lr_{:.6}_m_{}.pkl".
                       format(i+1,f1_score_value,accuracy_value,lr,m.name),"wb"))
        
        
    y_loss_validation = np.array(y_loss_validation)
    y_accuracy = np.array(y_accuracy)
    y_f1 = np.array(y_f1)
    x_axis = np.arange(epoch)
    
    
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
    


if __name__ == '__main__':
    # np.random.seed(0)
    
    train(0.2,30)
    
    lrs =[]
    
    # done
    # lrs.append(0.000001)
    # lrs.append(0.00001)
    # lrs.append(0.0001)
    
    # 
    # lrs.append(0.001)
    # lrs.append(0.003)
    # lrs.append(0.005)
    
    # 
    lrs.append(0.01)
    lrs.append(0.02)
    lrs.append(0.04)
    lrs.append(0.05)
    lrs.append(0.1)
    lrs.append(0.2)
    lrs.append(0.5)
    
    for lr in lrs:
        print("Learning rate: ",lr)
        train(lr,epoch=30)
    