"""Balanced accuracy.

From https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/metrics/_classification.py#L2111

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
"""
from .classifier_base import ClassifierBaseScoreType
from .macro_averaged_recall import MacroAveragedRecall


class BalancedAccuracy(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='balanced_accuracy', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        mac = MacroAveragedRecall()
        bac = mac(y_true_label_index, y_pred_label_index)

        return bac
