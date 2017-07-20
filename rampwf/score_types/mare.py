import numpy as np
from .base import BaseScoreType


class MARE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='mare', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true))
