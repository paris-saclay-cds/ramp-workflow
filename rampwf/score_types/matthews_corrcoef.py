from sklearn.metrics import matthews_corrcoef
from .classifier_base import ClassifierBaseScoreType


class MatthewsCorrcoef(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name='mcc', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = matthews_corrcoef(y_true_label_index, y_pred_label_index)
        return score
