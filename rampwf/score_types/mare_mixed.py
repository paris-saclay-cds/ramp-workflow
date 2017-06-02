from . import mare


def score_function(ground_truths, predictions, valid_indexes=None):
    """MARE of a mixed regression/classification prediction."""
    return mare.score_function(
        ground_truths.regression,
        predictions.regression,
        valid_indexes)

# default display precision in n_digits
precision = 2
