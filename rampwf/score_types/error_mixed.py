from rampwf.score_types import error


def score_function(ground_truths, predictions, valid_indexes=None):
    """Classification error of a mixed regression/classification prediction."""
    return error.score_function(
        ground_truths.multiclass_prediction,
        predictions.multiclass_prediction,
        valid_indexes)

# default display precision in n_digits
precision = 2
