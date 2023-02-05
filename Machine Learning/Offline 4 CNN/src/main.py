import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_folder = './../resource/NumtaDB_with_aug'
sub_folder = '/training-b/'
csv_file_name = base_folder + '/training-b.csv'

# read csv file
x = pd.read_csv(csv_file_name)
# iterate over rows with iterrows()
print("Total rows: {0}".format(len(x)))
x_max = 0
y_max = 0
for index, row in x.iterrows():
    # print(row['filename'])
    # print(row['digit'])
    image_fila_name = base_folder+sub_folder+row['filename']
    # read image using opencv or pillow
    
    if os.path.exists(image_fila_name):
        # print("File exists")
        img = cv2.imread(image_fila_name, cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        # show image
        # plt.imshow(img, cmap='gray', interpolation='bicubic')
        # plt.show()
        x_max = max(x_max, img.shape[0])
        y_max = max(y_max, img.shape[1])
        
        
    else:
        print("File does not exist:",image_fila_name)
    pass

print("x_max: {0}, y_max: {1}".format(x_max, y_max))
    