from sklearn.metrics import f1_score
from .classifier_base import ClassifierBaseScoreType


class F1(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='f1', precision=2, average="binary"):
        self.name = name
        self.precision = precision
        self.average = average

    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average=self.average)


class F1Micro(F1):
    def __init__(self, name="f1-micro", precision=2):
        F1.__init__(self, name=name, precision=precision, average="micro")
        
