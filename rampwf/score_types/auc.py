from sklearn.metrics import roc_auc_score


def score_function(ground_truths, predictions, valid_indexes=None):
    if valid_indexes is None:
        valid_indexes = range(len(ground_truths.y_pred))
    y_proba = predictions.y_pred[valid_indexes]
    y_true_proba = ground_truths.y_pred_label_index[valid_indexes]
    score = roc_auc_score(y_true_proba, y_proba[:, 1])
    return score

# default display precision in n_digits
precision = 2
