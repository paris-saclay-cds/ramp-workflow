from sklearn.metrics import accuracy_score
from .base import BaseScoreType


class ClassificationError(BaseScoreType):
    def __init__(self, name='classification error', precision=2, n_columns=2):
        self.name = name
        self.precision = precision
        # n_columns = 2: binary classification
        self.n_columns = n_columns
        self.is_lower_the_better = True
        self.minimum = 0.0,
        self.maximum = 1.0

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
        y_true_label_index = ground_truths.y_pred_label_index[valid_indexes]
        score = 1 - accuracy_score(y_true_label_index, y_pred_label_index)
        return score
