from . import classification_error


def score_function(ground_truths, predictions, valid_indexes=None):
    """Classification error of a mixed regression/classification prediction."""
    return classification_error.score_function(
        ground_truths.multiclass,
        predictions.multiclass,
        valid_indexes)

# default display precision in n_digits
precision = 2
