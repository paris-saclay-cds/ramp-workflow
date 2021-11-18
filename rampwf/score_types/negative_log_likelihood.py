from sklearn.metrics import log_loss

from .base import BaseScoreType


class NegativeLogLikelihood(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='negative log likelihood', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        score = log_loss(y_true_proba, y_proba)
        return score
