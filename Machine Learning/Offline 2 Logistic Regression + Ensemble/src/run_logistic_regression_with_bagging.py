"""
main code that you will run
"""

from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset
from metrics import Metrics

if __name__ == '__main__':
    # data load
    X, y = load_dataset()

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # training
    params = dict()
    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=9)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    metrics = Metrics(y_true=y_test, y_pred=y_pred)
    print('Accuracy ', metrics.accuracy)
    print('Recall score ', metrics.recall)
    print('Precision score ', metrics.precision)
    print('F1 score ', metrics.f1_score)