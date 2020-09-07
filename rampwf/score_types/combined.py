from .base import BaseScoreType


class Combined(BaseScoreType):
    is_lower_the_better = None
    minimum = 0.0
    maximum = 0.0

    def __init__(self, score_types, weights, name='combined', precision=2):
        self.name = name
        self.score_types = score_types
        self.weights = weights
        self.precision = precision
        for weight, score_type in zip(self.weights, self.score_types):
            if self.is_lower_the_better is None:
                self.is_lower_the_better = score_type.is_lower_the_better
            elif self.is_lower_the_better != score_type.is_lower_the_better:
                raise ValueError(
                    'Cannot combine scores of lower and higher the better')
            self.minimum += weight * score_type.minimum
            self.maximum += weight * score_type.maximum

    def score_function(self, ground_truths_combined, predictions_combined):
        score = 0.0
        for weight, score_type, ground_truths, predictions in zip(
                self.weights, self.score_types,
                ground_truths_combined.predictions_list,
                predictions_combined.predictions_list):
            score += weight * score_type.score_function(
                ground_truths, predictions)
        return score

    def __call__(self, y_true, y_pred):
        raise ValueError('Combined score has no deep score function.')
