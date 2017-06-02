import numpy as np
from .base import BaseScoreType


class RelativeRMSE(BaseScoreType):
    def __init__(self, name='rmse', precision=2, n_columns=0):
        self.name = name
        self.precision = precision
        self.n_columns = n_columns
        self.is_lower_the_better = True
        self.minimum = 0.0
        self.maximum = float('inf')

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_true = ground_truths.y_pred[valid_indexes]
        y_pred = predictions.y_pred[valid_indexes]
        self.check(y_true, y_pred)
        score = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
        return score
