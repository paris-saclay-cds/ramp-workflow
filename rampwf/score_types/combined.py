from .base import BaseScoreType


class Combined(BaseScoreType):
    is_lower_the_better = None
    is_rank_based = None
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
            if self.is_rank_based is None:
                self.is_rank_based = score_type.is_rank_based
            elif self.is_rank_based != score_type.is_rank_based:
                raise ValueError(
                    'Cannot combine rank-based and non-rank-based scores')
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
