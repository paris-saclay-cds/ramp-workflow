from . import mare_mixed
from . import error_mixed


def score_function(ground_truths, predictions, valid_indexes=None):
    return 2. / 3 * error_mixed.score_function(
        ground_truths, predictions, valid_indexes) +\
        1. / 3 * mare_mixed.score_function(
            ground_truths, predictions, valid_indexes)

# default display precision in n_digits
precision = 2
