import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k, X_train, y_train, x):

    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    distances = [(sqrt(np.sum((x_train - x)**2)), y_train[i])
                 for i, x_train in enumerate(X_train)]
    distances.sort()

    topK_y = [pair[1] for pair in distances[:k]]

    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]
