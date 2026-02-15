import numpy as np


def wmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    w = np.where(y_true > 0, 7.0, 1.0)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))
