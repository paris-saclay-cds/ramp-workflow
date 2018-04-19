import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import _check_targets

from .classifier_base import ClassifierBaseScoreType


def _balanced_accuracy_score(y_true, y_pred, sample_weight=None):
    """FIXME: port implementation of balanced accuracy from scikit-learn 0.20.
    """
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

class BalancedAccuracy(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='balanced_accuracy', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = _balanced_accuracy_score(y_true_label_index,
                                         y_pred_label_index)
        return score
