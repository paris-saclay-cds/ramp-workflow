import numpy as np


def get_score_cv_bags(Predictions, score_type, predictions_list, ground_truths,
                      test_is_list=None):
    """
    Compute the bagged scores of the predictions in predictions_list.

    test_is_list (list of list of integer indexes) controls which points
    in which fold participate in the combination. We return the
    combined predictions and a list of scores, where each element i is the
    score of the combination of the first i+1 folds.

    Parameters
    ----------
    Predictions : a class implementing BasePrediction signature
        Needed for the type of y_comb
    score_type : instance implementing BaseScoreType signature
    predictions_list : list of instances of Predictions
    ground_truths : instance of Predictions
    test_is_list : list of list of integers
        Indices of points that should be bagged in each prediction. If None,
        the full prediction vectors will be bagged.
    Returns
    -------
    combined_predictions : instance of Predictions
    score_cv_bags : list of scores (typically floats)
    """
    if test_is_list is None:  # we combine the full list
        test_is_list = [range(len(predictions.y_pred))
                        for predictions in predictions_list]

    y_comb = np.array(
        [Predictions(n_samples=len(ground_truths.y_pred))
         for _ in predictions_list])
    score_cv_bags = []
    for i, test_is in enumerate(test_is_list):
        # setting valid fold indexes of points to be combined
        y_comb[i].set_valid_in_train(predictions_list[i], test_is)
        # combine first i folds
        combined_predictions = Predictions.combine(y_comb[:i + 1])
        # get indexes of points with at least one prediction in
        # combined_predictions
        valid_indexes = combined_predictions.valid_indexes
        # score the combined predictions
        score_cv_bags.append(score_type.score_function(
            ground_truths, combined_predictions, valid_indexes))
        # Alex' old suggestion: maybe use masked arrays rather than passing
        # valid_indexes
    # TODO: will crash if len(test_is_list) == 0
    return combined_predictions, score_cv_bags
