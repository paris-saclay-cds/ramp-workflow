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
        return root_mean_squared_log_error(y_pred, y_true)
