from sklearn.metrics import cohen_kappa_score
from .classifier_base import ClassifierBaseScoreType


class Kappa(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name='kappa', precision=2, weights="quadratic"):
        self.name = name
        self.precision = precision
        self.weights = weights

    def __call__(self, y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights=self.weights)
