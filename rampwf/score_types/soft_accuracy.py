"""A generalization of the classification accuracy with cross-class scores.

Soften the accuracy score by giving scores through certain misclassifications
defined by the score matrix. For example, in ordinal regression we may want
not to penalize too much misclassifications to neighbor classes. The score also
generalizes RMSE-like regression scores for ordinal regression (when true and
predicted output levels are coming from a fixed set) by allowing to define
arbitrary misclassification scores.
"""

import numpy as np
from .base import BaseScoreType


class SoftAccuracy(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0

    def __init__(self, score_matrix, name='soft precision', precision=2):
        self.name = name
        self.precision = precision
        self.maximum = np.max(score_matrix)
        self.score_matrix = score_matrix

    def __call__(self, y_true_proba, y_proba):
        # Clip negative probas
        y_proba_positive = np.clip(y_proba, 0, 1)
        # normalize rows
        y_proba_normalized = y_proba_positive / np.sum(
            y_proba_positive, axis=1, keepdims=True)
        # Smooth true probabilities with score_matrix
        y_true_smoothed = y_true_proba.dot(self.score_matrix)
        # Compute dot product between the predicted probabilities and
        # the smoothed true "probabilities" ("" because it does not sum to 1)
        scores = np.sum(y_proba_normalized * y_true_smoothed, axis=1)
        scores = np.nan_to_num(scores)
        score = np.mean(scores)
        # to pick up all zero probabilities
        score = np.nan_to_num(score)
        return score
