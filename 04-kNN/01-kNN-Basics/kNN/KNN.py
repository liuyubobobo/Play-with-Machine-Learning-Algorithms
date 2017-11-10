import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k, train_X, train_y, x):
    """
    kNN分类算法
    :param k: 最近邻算法中的k
    :param train_X: 训练数据集X，为二维矩阵
    :param train_y: 训练数据集对应的标签y，为一维向量
    :param x: 待预测的数据，为一维向量
    :return: kNN算法给出的带预测数据x对应的标签
    """

    assert train_X.shape[0] == train_y.shape[0]
    assert train_X.shape[1] == x.shape[0]

    distances = [(sqrt(((train_x - x)**2).sum()), train_y[i])
                 for i, train_x in enumerate(train_X)]
    distances.sort()

    topk_y = [pair[1] for pair in distances[:k]]

    votes = Counter(topk_y)
    return votes.most_common(1)[0][0]
