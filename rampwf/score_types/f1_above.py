from sklearn.metrics import f1_score
from .classifier_base import ClassifierBaseScoreType


class F1Above(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1_above', precision=2, threshold=0.5):
        self.name = name
        self.precision = precision
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average=None)
        return 1. * len(f1[f1 > self.threshold]) / len(f1)
