import numpy as np
from sklearn import preprocessing


def standardize_matrix(X):
    """ """
    num_cols = X.shape[1]
    norm_matrix = np.zeros(X.shape)

    for col_num in range(num_cols):
        X_col = X[:, col_num]
        norm_matrix[:, col_num] = (X_col - np.mean(X_col)) / np.std(X_col)

    # scaler = preprocessing.StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)

    return norm_matrix
