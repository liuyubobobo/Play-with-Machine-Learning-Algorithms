import numpy as np


def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum([1 if y_true[i] == y_predict[i] else 0 for i in range(len(y_true))]) / len(y_true)
