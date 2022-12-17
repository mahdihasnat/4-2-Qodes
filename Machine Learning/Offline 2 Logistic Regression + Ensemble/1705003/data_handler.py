import numpy as np

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
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
    assert X.shape[0] == y.shape[0], "X and y have different number of rows"
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
    
    test_length = int(X.shape[0] * test_size)
    train_length = X.shape[0] - test_length
    X_train,X_test = np.split(X, [train_length], axis=0)
    y_train,y_test = np.split(y, [train_length], axis=0)
    
    
    # # adding noise data
    # noise_data_size = round(X_train.shape[0]*1.0)
    # print("noise data size ", noise_data_size)
    # # for each feature : calc min and max and then generate noise data
    # X_noise = np.zeros((noise_data_size, X_train.shape[1]))
    # for i in range(X_train.shape[1]):
    #     min = np.min(X_train[:,i])
    #     max = np.max(X_train[:,i])
    #     X_noise[:,i] = np.random.uniform(min, max, noise_data_size)
    
    # y_noise = np.zeros((noise_data_size, 1))
    # y_noise[:,0] = np.random.randint(0, 2, noise_data_size)
    
    # X_train = np.concatenate((X_train, X_noise), axis=0)
    # y_train = np.concatenate((y_train, y_noise), axis=0)  
    
    
    
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    
    indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
    X_sample = X[indices]
    y_sample = y[indices]
    
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    
    return X_sample, y_sample
