from sklearn.metrics import accuracy_score
from .classifier_base import ClassifierBaseScoreType


class ClassificationError(ClassifierBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='classification error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        return 1 - accuracy_score(y_true_label_index, y_pred_label_index)
