import time
from linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import accuracy, precision_score, recall_score, f1_score, Metrics

if __name__ == '__main__':
    start_time = time.time()
    # data load
    X, y = load_dataset()
    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, test_size=0.2, shuffle=True)
    # print("X_train ", X_train)
    # print("y_train ", y_train)
    # print("X_test ", X_test)
    # print("y_test ", y_test)
    
    # training
    params = dict()
    params['alpha'] = 0.01
    params['max_iter'] = 100000
    
    classifier = LogisticRegression(params)
    # classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)
    
    # print("shape of y_pred ", y_pred.shape)
    # print("shape of y_test ", y_test.shape)
    
    
    # performance on test set
    metrics = Metrics(y_true=y_test, y_pred=y_pred)
    print('Accuracy ', metrics.accuracy)
    print('Recall score ', metrics.recall)
    print('Precision score ', metrics.precision)
    print('F1 score ', metrics.f1_score)
    print("Time taken ", time.time() - start_time)