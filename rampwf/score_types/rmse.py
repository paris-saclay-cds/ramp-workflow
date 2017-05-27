import numpy as np


def score_function(ground_truths, predictions, valid_indexes=None):
    if valid_indexes is None:
        valid_indexes = slice(None, None, None)
    y_pred = predictions.y_pred[valid_indexes]
    y_true = ground_truths.y_pred[valid_indexes]
    score = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return score

# default display precision in n_digits
precision = 2
