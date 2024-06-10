from .base import BaseScoreType

from sklearn.metrics import root_mean_squared_log_error


class RMSLE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="rmsle", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        # sklearn is raising an error if y_pred or y_true are negative.
        # we force negative predictions to be 0. we do not touch the true targets
        # so the user knows he should not be using this score for negative targets.
        neg_ind = y_pred < 0
        y_pred[neg_ind] = 0
        neg_ind = y_true < 0
        y_true[neg_ind] = 0
        return root_mean_squared_log_error(y_true, y_pred)
