from sklearn.metrics import r2_score
from .base import BaseScoreType


class R2(BaseScoreType):
    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name="r2", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
