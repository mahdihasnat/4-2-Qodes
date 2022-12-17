from data_handler import bagging_sampler
import copy
import numpy as np
class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0] , "X and y must have the same number of rows"
        assert len(X.shape) == 2, "X must be a 2D array"
        self.estimators = []
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            new_estimator = copy.copy(self.base_estimator)
            new_estimator.fit(X_sample, y_sample)
            self.estimators.append(new_estimator)
        
        
    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        result = 0
        for i in range(self.n_estimator):
            result += self.estimators[i].predict(X)
        result = result / self.n_estimator
        return np.where(result > 0.5, 1, 0)