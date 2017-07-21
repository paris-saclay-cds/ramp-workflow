from sklearn.metrics import accuracy_score
from .classifier_base import ClassifierBaseScoreType


class Accuracy(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='accuracy', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = accuracy_score(y_true_label_index, y_pred_label_index)
        return score
