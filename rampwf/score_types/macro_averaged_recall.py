from sklearn.metrics import recall_score
from .classifier_base import ClassifierBaseScoreType


class MacroAveragedRecall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='accuracy', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = recall_score(
            y_true_label_index, y_pred_label_index, average='macro')
        return score
