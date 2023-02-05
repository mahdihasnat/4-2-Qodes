import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Required magic to display matplotlib plots in notebooks
# %matplotlib inline

def load_dataset():
	base_folder = './../resource/NumtaDB_with_aug'
	sub_folder = '/training-b/'
	csv_file_name = base_folder + '/training-b.csv'
	csv = pd.read_csv(csv_file_name)
	
	print("Total rows: {0}".format(csv.shape[0]))
	x_max = 0
	y_max = 0
	image_shape = (128,128)
	channel = 1
 
	x,y = np.array([]), np.array([])
	x.reshape((0,channel,image_shape[0],image_shape[1]))
	y.reshape(0,1)
	print("x shape: {0}".format(x.shape))
	print("y shape: {0}".format(y.shape))
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
			img/=255.0
			# print(img.dtype)
			print("Shape of img",img.shape)
   
			print("shape of x",x.shape)
			print("shape of y",y.shape)
   
			x=np.append(x, img, axis= 0)
			y=np.append(y, row['digit'], axis= 0)
			# print(img)
			print("shape of x",x.shape)
			print("shape of y",y.shape)
			break
		else:
			print("File does not exist:",image_fila_name)
		pass

	# print("x_max: {0}, y_max: {1}".format(x_max, y_max))
	print("image_shape: {0}".format(image_shape))

load_dataset()