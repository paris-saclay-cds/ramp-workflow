from sklearn.metrics import roc_auc_score
from .base import BaseScoreType


class ROCAUC(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='roc_auc', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions):
        """A hybrid score.

        It tests the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        true probability of the second class). Thus we have to override the
        `Base` function here
        """
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        return roc_auc_score(y_true_proba, y_proba)
