import time
from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import Metrics

if __name__ == '__main__':
    start_time = time.time()
    
    # data load
    X, y = load_dataset()
    test_size = 0.9
    shuffle = True
    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, 
                                test_size=test_size, shuffle=shuffle)

    print("shape of X_train ", X_train.shape)
    print("shape of X_test ", X_test.shape)
    
    # training
    params = dict()
    params['alpha'] = 0.01
    params['max_iter'] = 1000
    n_estimator = 15
    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=n_estimator)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

     # performance on test set
    metrics = Metrics(y_true=y_test, y_pred=y_pred)
    print('Test set size ', test_size)
    print('Shuffle ', shuffle)
    print("Number of iterations ", params['max_iter'])
    print("Learning rate ", params['alpha'])
    print("Number of estimators ", n_estimator)
    print("----------------------------------")
    print('Accuracy ', metrics.accuracy)
    print('Recall score ', metrics.recall)
    print('Precision score ', metrics.precision)
    print('F1 score ', metrics.f1_score)
    print("Time taken ", time.time() - start_time)