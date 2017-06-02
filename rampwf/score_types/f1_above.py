from sklearn.metrics import f1_score
from .base import BaseScoreType


class F1Above(BaseScoreType):
    def __init__(self, name='f1_above', precision=2, n_columns=2,
                 threshold=0.5):
        self.name = name
        self.precision = precision
        # n_columns = 2: binary classification
        self.n_columns = n_columns
        self.is_lower_the_better = False
        self.minimum = 0.0
        self.maximum = 1.0
        self.threshold = threshold

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        """Rate of classes with f1 score above threshold."""
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_pred = predictions.y_pred_label_index[valid_indexes]
        y_true = ground_truths.y_pred_label_index[valid_indexes]
        f1 = f1_score(y_true, y_pred, average=None)
        score = 1. * len(f1[f1 > self.threshold]) / len(f1)
        return score
