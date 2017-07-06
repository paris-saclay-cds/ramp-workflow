from sklearn.metrics import accuracy_score
from .base import BaseScoreType


class Accuracy(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='accuracy', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
        y_true_label_index = ground_truths.y_pred_label_index[valid_indexes]
        self.check_y_pred_dimensions(y_true_label_index, y_pred_label_index)
        score = accuracy_score(y_true_label_index, y_pred_label_index)
        return score
