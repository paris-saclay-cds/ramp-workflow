import numpy as np
from .base import BaseScoreType


class CyclicRMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse', precision=2, periodicity=-1):
        self.name = name
        self.precision = precision
        self.periodicity = -1

    def __call__(self, y_true, y_pred):
        d = y_true - y_pred
        if(self.periodicity > 0):
            d = min(np.mod(d, self.periodicity),
                    np.mod(-d, self.periodicity))
        return np.sqrt(np.mean(np.square(d)))
