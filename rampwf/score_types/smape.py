import numpy as np

from .base import BaseScoreType


class SMAPE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="smape", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        apes = np.zeros(len(y_true))
        zero_mask = np.logical_and(y_true == 0, y_pred == 0).ravel()
        y_true_ = y_true[~zero_mask].ravel()
        y_pred_ = y_pred[~zero_mask].ravel()
        apes[~zero_mask] = (
            2 * np.abs(y_true_ - y_pred_) / (np.abs(y_true_) + np.abs(y_pred_))
        )
        return np.mean(apes) * 100
