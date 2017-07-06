import numpy as np
from .base import BaseScoreType


class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_true = ground_truths.y_pred[valid_indexes]
        y_pred = predictions.y_pred[valid_indexes]
        self.check_y_pred_dimensions(y_true, y_pred)
        score = np.sqrt(np.mean(np.square(y_true - y_pred)))
        return score
