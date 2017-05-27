from sklearn.metrics import f1_score


def score_function(ground_truths, predictions, valid_indexes=None):
    """Rate of classes with f1 score above 0.5."""
    if valid_indexes is None:
        valid_indexes = slice(None, None, None)
    y_pred = predictions.y_pred_label_index[valid_indexes]
    y_true = ground_truths.y_pred_label_index[valid_indexes]
    f1 = f1_score(y_true, y_pred, average=None)
    score = 1. * len(f1[f1 > 0.5]) / len(f1)
    return score

# default display precision in n_digits
precision = 2
