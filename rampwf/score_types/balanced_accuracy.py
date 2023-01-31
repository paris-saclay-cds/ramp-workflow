"""Balanced accuracy.

From https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/\
    metrics/_classification.py#L2111

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class. With the use of the parameter 'adjusted', 
    balanced accuracy can be adjusted between 1/(1-nclasses) and 1.
"""
from .classifier_base import ClassifierBaseScoreType
from sklearn.metrics import balanced_accuracy_score


class BalancedAccuracy(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='balanced_accuracy', precision=2, adjusted=True):
        self.name = name
        self.precision = precision
        self.adjusted = adjusted

    def __call__(self, y_true_label_index, y_pred_label_index):
        """
        When adjusted = True, it will use an adjusted balanced_accuracy_score
        from sklearn which is calculated by subtracting the base true positive
        rate (i.e. the chance recall) from the macro averaged recall, and dividing
        the result by (1 - base true positive rate). Score will then be between
        1 / (1 - nclasses) and 1.
        When adjusted = False, it will use the non-adjusted balanced_accuracy_score
        from sklearn.
        """
        score = balanced_accuracy_score(
            y_true_label_index, y_pred_label_index, adjusted=self.adjusted)

        return score
