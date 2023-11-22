import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 123


def load_data():
    """ """

    data = load_iris()
    X = data["data"]
    y = data["target"]

    # print(data["target_names"])

    return X, y


def split_data(X, y, test_size=0.2, seed=SEED):
    """ """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for _, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test
