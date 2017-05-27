from sklearn.metrics import accuracy_score


def score_function(ground_truths, predictions, valid_indexes=None):
    if valid_indexes is None:
        valid_indexes = slice(None, None, None)
    y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
    y_true_label_index = ground_truths.y_pred_label_index[valid_indexes]
    score = accuracy_score(y_true_label_index, y_pred_label_index)
    return score

# default display precision in n_digits
precision = 2
