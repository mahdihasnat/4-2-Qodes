import numpy as np
import pandas as pd
import cv2

base_folder = './../resource/NumtaDB_with_aug'
sub_folder = '/testing-b/'
csv_file_name = base_folder + '/training-b.csv'

# read csv file
x = pd.read_csv(csv_file_name)
# iterate over rows with iterrows()
print("Total rows: {0}".format(len(x)))
for index, row in x.iterrows():
    # print(row['filename'])
    # print(row['digit'])
    image_fila_name = base_folder+sub_folder+row['filename']
    # read image using opencv or pillow
    img = cv2.imread(image_fila_name, cv2.IMREAD_GRAYSCALE)
    print(type(img))
    pass
    