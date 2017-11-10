import numpy as np

class KNN:

    def __init__(self, k):
        '''
        初始化kNN算法，传入参数k
        :param k:  
        '''
        self.k = k

    def fit(self, X, y):
        '''
        
        :param X: 
        :param y: 
        :return: 
        '''
        self._fit_X = X.copy()
        self._fit_y = y.copy()
        return self

    def predict(self, x):
        pass