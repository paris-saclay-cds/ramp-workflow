import rampwf.score_types.mare_mixed as mare
import rampwf.score_types.error_mixed as error


def score_function(ground_truths, predictions, valid_indexes=None):
    return 2. / 3 * error.score_function(
        ground_truths, predictions, valid_indexes) +\
        1. / 3 * mare.score_function(
            ground_truths, predictions, valid_indexes)

# default display precision in n_digits
precision = 2
