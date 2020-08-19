import numpy as np
from .classifier_base import ClassifierBaseScoreType


def gini(y_true, y_pred, cmpcol=0, sortcol=1):
    all = np.asarray(
        np.c_[y_true, y_pred, np.arange(len(y_true))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -all[:, 1]))]
    total_losses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / total_losses
    giniSum -= (len(y_true) + 1) / 2.
    return giniSum / len(y_true)


class NormalizedGini(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name='normalized_gini', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions):
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true, y_pred):
        return gini(y_true, y_pred) / gini(y_true, y_true)
