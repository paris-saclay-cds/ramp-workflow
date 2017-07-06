from sklearn.metrics import roc_auc_score
from .base import BaseScoreType


class ROCAUC(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='roc_auc', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_proba = predictions.y_pred[valid_indexes]
        y_true_proba = ground_truths.y_pred_label_index[valid_indexes]
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        score = roc_auc_score(y_true_proba, y_proba[:, 1])
        return score
