import numpy as np


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100
