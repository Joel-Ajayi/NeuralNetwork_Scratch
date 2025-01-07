import numpy as np


def binary_cross_entropy(AL, Y):
    # Y-Labelled Data of correct category
    # AL-Nueral Network Predicted category
    eps = 1e-100
    log_probs = np.multiply(-np.log(np.clip(AL, eps, 1 - eps)), Y) + np.multiply(
        -np.log(np.clip(1 - AL, eps, 1 - eps)), 1 - Y
    )
    return np.sum(log_probs) / Y.shape[0]


def category_cross_entropy(AL, Y):
    # Y-Labelled Data of correct category
    # AL-Nueral Network Predicted category
    eps = 1e-100
    return -np.sum(np.multiply(Y, np.log(np.clip(AL, eps, 1.0 - eps))))
