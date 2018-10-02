import numpy as np
from .base import BaseScoreType


class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
