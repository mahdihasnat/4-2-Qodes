def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    # CSV reader 
    # import csv
    filename = 'data_banknote_authentication.csv'
    # file = open(filename, 'r')
    # csvreader = csv.reader(file)
    # print(next(csvreader))
    
    import numpy as np
    data = np.genfromtxt(filename, delimiter=',', skip_header=1,dtype=np.float64)
    # print(data.shape)
    data1 , data2 = np.split(data, [4], axis=1)
    # print(data1.shape)
    # print(data2.shape)
    
    return data1, data2


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train, y_train, X_test, y_test = None, None, None, None
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
