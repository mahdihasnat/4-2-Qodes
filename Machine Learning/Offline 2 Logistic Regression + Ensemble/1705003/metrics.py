import numpy as np

"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""


class Metrics:
    def __init__(self , y_true, y_pred) -> None :
        assert y_true.shape == y_pred.shape , "y_true and y_pred have different shape"
        assert y_true.shape[1] == 1 , "y_true has multiple columns"
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for y_t, y_p in np.nditer([y_true, y_pred]):
            if np.equal(y_p ,1):
                tp += y_t
                fp += 1-y_t
            else:
                fn += y_t
                tn += 1-y_t
        self.accuracy = (tp+tn) / y_true.shape[0]
        self.recall = tp / (tp + fn)
        self.precision = tp / (tp + fp)
        self.f1_score = 2 * tp / (2 * tp + fp + fn)
    

def accuracy(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: accuracy = (tp + tn) / (tp + tn + fp + fn)
    """
    # todo: implement
    assert y_true.shape == y_pred.shape , "y_true and y_pred have different shape"
    assert y_true.shape[1] == 1 , "y_true has multiple columns"
    tp_plus_tn = 0
    for y_t, y_p in np.nditer([y_true, y_pred]):
        if y_t == y_p:
            tp_plus_tn += 1
    return tp_plus_tn / y_true.shape[0]   
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return: precision = tp / (tp + fp)
    """
    assert y_true.shape == y_pred.shape , "y_true and y_pred have different shape"
    assert y_true.shape[1] == 1 , "y_true has multiple columns"
    tp = 0
    fp = 0
    for y_t, y_p in np.nditer([y_true, y_pred]):
        if np.equal(y_p, 1):
            tp+=y_t
            fp+=1-y_t
    
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return: recall = tp / (tp + fn)
    """
    assert y_true.shape == y_pred.shape , "y_true and y_pred have different shape"
    assert y_true.shape[1] == 1 , "y_true has multiple columns"

    tp = 0
    fn = 0
    for y_t, y_p in np.nditer([y_true, y_pred]):
        if np.equal(y_t, 1):
            tp+=y_p
            fn+=1-y_p
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return: f1 score = 2 * (precision * recall) / (precision + recall) = 2 * tp / (2 * tp + fp + fn)
    """
    assert y_true.shape == y_pred.shape , "y_true and y_pred have different shape"
    assert y_true.shape[1] == 1 , "y_true has multiple columns"
    
    tp = 0
    fp = 0
    fn = 0
    for y_t, y_p in np.nditer([y_true, y_pred]):
        if np.equal(y_p ,1):
            tp+=y_t
            fp+=1-y_t
        else:
            fn+=y_t
    
    return 2 * tp / (2 * tp + fp + fn)
