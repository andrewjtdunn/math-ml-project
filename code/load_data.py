import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

SEED = 123


def load_data():
    """ """

    data = load_iris()
    X = data["data"]
    y = data["target"]

    print(data["target_names"])

    return X, y


def split_data(X, y, seed=SEED):
    """ """
    skf = StratifiedKFold(random_state=seed)

    index_1 = skf.split(X[:50], y[:50])
    indices = index_1 + index_1 * 2 + index_1 * 3

    train_idx, test_idx = indices

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
