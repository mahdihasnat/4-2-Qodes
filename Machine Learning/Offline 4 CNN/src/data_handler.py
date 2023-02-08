import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import os
# Required magic to display matplotlib plots in notebooks
# %matplotlib inline


def get_dataset(image_shape,channel,sample_bound,base_folder,csv_file_name):
    csv = pd.read_csv(os.path.join(base_folder,csv_file_name))
    print("Total rows: {0}".format(csv.shape[0]))
    
    x_list = []
    y_list = []
    
    for index, row in csv.iterrows():
        # print(row['filename'])
        # print(row['digit'])
        # print(row['database name'])
        subfolder = row['database name']
        image_fila_name = os.path.join(base_folder,subfolder,row['filename'])
        # # read image using opencv or pillow
        
        if os.path.exists(image_fila_name):
            # print("File exists")
            img = cv2.imread(image_fila_name, cv2.IMREAD_GRAYSCALE)
            # print(img.shape)
            # show image
            
            img = cv2.resize(img, image_shape)
            # print(img.dtype)
            # print(img.shape)
            # add 1 channel dimension
            # print(img.dtype)
            # print(img.shape)
            # print(img)
            # dilute image
            
            img = 255-img
            # plt.imshow(img, cmap='gray', interpolation='bicubic')
            # plt.show()
            cv2.dilate(img, np.ones((10,10),np.uint8), iterations = 1)
            # plt.imshow(img, cmap='gray', interpolation='bicubic')
            # plt.show()
            
            img = np.expand_dims(img, axis=0)
            
            img=img.astype(np.float32)
            # print(img.dtype)
            # img = img/np.maximum(img.max(),1)
            img /= 255
            # print(img.dtype)
            # print("Shape of img",img.shape)
   
            # print("shape of x",x.shape)
            # print("shape of y",y.shape)
            x_list.append(img)
            y_np = np.zeros((10,))
            y_np[row['digit']] = 1
            y_list.append(np.array(y_np))
            
            # print(img)
            # print("shape of x",x.shape)
            # print("shape of y",y.shape)
            # break
            if sample_bound != -1 and len(x_list) >= sample_bound:
                break
        else:
            print("File does not exist:",image_fila_name)
        pass

    return x_list,y_list

def load_dataset(image_shape=(28,28),sample_bound=-1):
    base_folder = './../resource/NumtaDB_with_aug'
    x_list = []
    y_list = []
    channel = 1
    
    
    csv_file_name = 'training-b.csv'
    tx,ty= get_dataset(
                       image_shape=image_shape,
                       channel = channel,
                       sample_bound=sample_bound,
                       base_folder=base_folder,
                       csv_file_name=csv_file_name
                       )
    x_list.extend(tx)
    y_list.extend(ty)
    
    csv_file_name ='training-a.csv'
    tx,ty= get_dataset(
                       image_shape=image_shape,
                       channel = channel,
                       sample_bound=sample_bound,
                       base_folder=base_folder,
                       csv_file_name=csv_file_name
                       )
    x_list.extend(tx)
    y_list.extend(ty)
    
    csv_file_name = 'training-c.csv'
    tx,ty= get_dataset(
                       image_shape=image_shape,
                       channel = channel,
                       sample_bound=sample_bound,
                       base_folder=base_folder,
                       csv_file_name=csv_file_name
                       )
    x_list.extend(tx)
    y_list.extend(ty)
    
    
    
    x=np.array(x_list)
    y=np.array(y_list)
    # print("x_max: {0}, y_max: {1}".format(x_max, y_max))
    print("image_shape: {0}".format(image_shape))
    print("x shape: {0}".format(x.shape))
    print("y shape: {0}".format(y.shape))
    assert x.shape[0] == y.shape[0], "x and y have different number of rows"
    assert x.shape[1] == channel, "x has different number of channels"
    assert x.shape[2] == image_shape[0], "x has different height"
    assert x.shape[3] == image_shape[1], "x has different width"
    assert y.shape[1] == 10, "y has different number of classes"
    assert len(x.shape) == 4, "x shape is not 4D"
    assert len(y.shape) == 2, "y shape is not 2D"
    
    return x,y