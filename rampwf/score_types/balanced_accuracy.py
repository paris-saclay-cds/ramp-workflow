"""Balanced accuracy.

From https://github.com/ch-imad/AutoMl_Challenge/blob/2353ec0/Starting_kit/scoring_program/libscores.py#L187  # noqa

See the thread at 
https://github.com/rhiever/tpot/issues/108#issuecomment-317067760
about the different definitions.
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
        tpr = mac(y_true_label_index, y_pred_label_index)
        base_tpr = 1. / len(self.label_names)
        score = (tpr - base_tpr) / (1 - base_tpr)
        return score
