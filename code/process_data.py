import numpy as np


def normalize_X_matrix(X):
    """ """
    num_cols = X.shape[1]
    norm_matrix = np.zeroes(X.shape)

    for col in range(num_cols):
        norm_matrix[:, col] = np.linalg.norm(X[:, col])

    return norm_matrix
