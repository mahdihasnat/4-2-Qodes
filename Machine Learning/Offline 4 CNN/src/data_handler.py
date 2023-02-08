import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Required magic to display matplotlib plots in notebooks
# %matplotlib inline

def load_dataset(image_shape=(28,28),sample_bound=-1):
    base_folder = './../resource/NumtaDB_with_aug'
    sub_folder = '/training-a/'
    csv_file_name = base_folder + '/training-a.csv'
    csv = pd.read_csv(csv_file_name)
    
    print("Total rows: {0}".format(csv.shape[0]))
    x_max = 0
    y_max = 0
    
    channel = 1
    x_list = []
    y_list = []
    for index, row in csv.iterrows():
        # print(row['filename'])
        # print(row['digit'])
        image_fila_name = base_folder+sub_folder+row['filename']
        # read image using opencv or pillow
        
        if os.path.exists(image_fila_name):
            # print("File exists")
            img = cv2.imread(image_fila_name, cv2.IMREAD_GRAYSCALE)
            # print(img.shape)
            # show image
            x_max = max(x_max, img.shape[0])
            y_max = max(y_max, img.shape[1])
            img = cv2.resize(img, image_shape)
            # plt.imshow(img, cmap='gray', interpolation='bicubic')
            # plt.show()
            # print(img.dtype)
            # print(img.shape)
            # add 1 channel dimension
            img = np.expand_dims(img, axis=0)
            # print(img.dtype)
            # print(img.shape)
            # print(img)
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
    # sz = min(5,len(x_list))
    # x_list = x_list[:sz]
    # y_list = y_list[:sz]
    # convert x_list to numpy array
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
