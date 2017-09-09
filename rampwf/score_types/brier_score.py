from .base import BaseScoreType
import numpy as np


class BrierScore(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='brier_score', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        """A hybrid score.

        It tests the the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        tru probabilty of the second class). Thus we have to override the
        `Base` function here
        """
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_proba = predictions.y_pred[valid_indexes][:, 1]
        y_true_proba = ground_truths.y_pred_label_index[valid_indexes]
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        return np.mean((y_proba - y_true_proba) ** 2)
