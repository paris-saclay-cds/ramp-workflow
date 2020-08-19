from .base import BaseScoreType


class MakeCombined(BaseScoreType):
    def __init__(self, score_type, index):
        self.score_type = score_type
        self.is_lower_the_better = score_type.is_lower_the_better
        self.minimum = score_type.minimum
        self.maximum = score_type.maximum
        self.name = score_type.name
        self.precision = score_type.precision
        self.index = index

    def score_function(self, ground_truths_combined, predictions_combined):
        return self.score_type.score_function(
            ground_truths_combined.predictions_list[self.index],
            predictions_combined.predictions_list[self.index])

    def __call__(self, y_true, y_pred):
        raise ValueError('MakeCombined score has no deep score function.')
