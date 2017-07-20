import numpy as np
from .base import BaseScoreType


class NegativeLogLikelihood(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='negative lof likelihood', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        # Normalize rows
        y_proba_normalized = y_proba / np.sum(y_proba, axis=1, keepdims=True)
        # Kaggle's rule
        y_proba_normalized = np.maximum(y_proba_normalized, 10 ** -15)
        y_proba_normalized = np.minimum(y_proba_normalized, 1 - 10 ** -15)
        scores = - np.sum(np.log(y_proba_normalized) * y_true_proba, axis=1)
        score = np.mean(scores)
        return score
