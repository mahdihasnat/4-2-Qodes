import numpy as np

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    filename = 'data_banknote_authentication.csv'
    
    data = np.genfromtxt(filename, delimiter=',', skip_header=1,dtype=np.float64)
    data1 , data2 = np.split(data, [4], axis=1)
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
    assert X.shape[0] == y.shape[0], "X and y have different number of rows"
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
    
    test_length = int(X.shape[0] * test_size)
    train_length = X.shape[0] - test_length
    X_train,X_test = np.split(X, [train_length], axis=0)
    y_train,y_test = np.split(y, [train_length], axis=0)
    # X_train, y_train, X_test, y_test = None, None, None, None
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
