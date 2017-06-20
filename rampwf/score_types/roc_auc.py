from sklearn.metrics import roc_auc_score
from .base import BaseScoreType


class ROCAUC(BaseScoreType):
    def __init__(self, name='roc_auc', precision=2, n_columns=2):
        self.name = name
        self.precision = precision
        # n_columns = 2: binary classification
        self.n_columns = n_columns
        self.is_lower_the_better = False
        self.minimum = 0.0
        self.maximum = 1.0

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_proba = predictions.y_pred[valid_indexes]
        y_true_proba = ground_truths.y_pred_label_index[valid_indexes]
        score = roc_auc_score(y_true_proba, y_proba[:, 1])
        return score
