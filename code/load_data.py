import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from preprocess import load_shelter_data

SEED = 123


def load_data(param):
    """
    """
    if param == 'iris':
        data = load_iris()
        X = data["data"]
        y = data["target"]
    
    if param == 'austin_shelter_cats':
        data = load_shelter_data()
        X = data[:, 1:]
        y = data[:, 0]

    return X, y


def split_data(X, y, test_size=0.2, seed=SEED):
    """ """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for _, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # added column of 1's to train and test for bias
    X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis = 1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis = 1)

    return X_train, X_test, y_train, y_test
