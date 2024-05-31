import numpy as np
from sklearn.metrics import median_absolute_error
from .base import BaseScoreType


class MedAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="medae", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return median_absolute_error(y_true, y_pred)
