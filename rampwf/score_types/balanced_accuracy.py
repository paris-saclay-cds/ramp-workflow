from .classifier_base import ClassifierBaseScoreType
from sklearn.metrics import recall_score
from sklearn.metrics.classification import _check_targets


def _balanced_accuracy_score(y_true, y_pred, sample_weight=None):
    """FIXME: port implementation of balanced accuracy from scikit-learn 0.20.
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type != 'binary':
        raise ValueError('Balanced accuracy is only meaningful '
                         'for binary classification problems.')
    # simply wrap the ``recall_score`` function
    return recall_score(y_true, y_pred,
                        pos_label=None,
                        average='macro',
                        sample_weight=sample_weight)


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
