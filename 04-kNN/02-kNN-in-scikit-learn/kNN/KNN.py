import numpy as np

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, train_X, train_y):
        self._train_X = train_X
        self._train_y = train_y
        return self

    def predict(self, test_X):
        '''
        
        :param test_X: 
        :return: 
        '''
        pass