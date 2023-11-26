import numpy as np
from sklearn import preprocessing


def standardize_matrix(X):
    """ """
    num_cols = X.shape[1]
    norm_matrix = np.zeros(X.shape)

    for col_num in range(num_cols):
        X_col = X[:, col_num]
        col_std = np.std(X_col)

        if col_std > 1e-8:
            norm_matrix[:, col_num] = (X_col - np.mean(X_col)) / col_std

        # if there's no standard deviation, just subtract the mean
        else:
            norm_matrix[:, col_num] = X_col - np.mean(X_col)

    # scaler = preprocessing.StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)

    return norm_matrix
