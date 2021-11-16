import numpy as np
import copy


def combine_predictions(Predictions, predictions_list, index_list):
    """Combine predictions predictions_list[index_list].

    Parameters
    ----------
    Predictions : a Predictions type
        Needed to call combine.
    predictions_list : list of instances of Predictions
    index_list : list of integers
        Indices of the submissions to combine (possibly with replacement).

    Returns
    -------
    combined_predictions : instance of Predictions
    """
    predictions_list_to_combine = [predictions_list[i] for i in index_list]
    combined_predictions = Predictions.combine(predictions_list_to_combine)
    return combined_predictions


def get_score_cv_bags(score_type, predictions_list, ground_truths,
                      test_is_list=None):
    """
    Compute the bagged scores of the predictions in predictions_list.

    test_is_list (list of list of integer indexes) controls which points
    in which fold participate in the combination. We return the
    combined predictions and a list of scores, where each element i is the
    score of the combination of the first i+1 folds.

    Parameters
    ----------
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
    Predictions = type(ground_truths)
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
        combined_predictions = combine_predictions(
            Predictions, y_comb[:i + 1], range(i + 1))
        # get indexes of points with at least one prediction in
        # combined_predictions
        valid_indexes = combined_predictions.valid_indexes
        # set valid slices in ground truth and predictions
        ground_truths_local = copy.deepcopy(ground_truths)
        ground_truths_local.set_slice(valid_indexes)
        combined_predictions.set_slice(valid_indexes)
        # score the combined predictions
        score_of_prefix = score_type.score_function(
            ground_truths_local, combined_predictions)
        score_cv_bags.append(score_of_prefix)
        # Alex' old suggestion: maybe use masked arrays rather than passing
        # valid_indexes
    # TODO: will crash if len(test_is_list) == 0
    return combined_predictions, score_cv_bags


def _get_next_best_submission(predictions_list, ground_truths,
                              score_type, best_index_list,
                              min_improvement=0.0):
    """Find net best submission if added to predictions_list[best_index_list].

    Find the model that minimizes the score if added to
    predictions_list[best_index_list] using score_type.score_function.
    If there is no model improving the input
    combination, the input best_index_list is returned. Otherwise the index of
    the best model is added to the list. We could also return the combined
    prediction (for efficiency, so the combination would not have to be done
    each time; right now the algo is quadratic), but I don't think any
    meaningful rule will be associative, in which case we should redo the
    combination from scratch each time the set changes. Since mostly
    combination = mean, we could maintain the sum and the number of models, but
    it would be a bit bulky. We'll see how this evolves.

    Parameters
    ----------
    predictions_list : list of instances of Predictions
        Each element of the list is an instance of Predictions of a model
        on the same (cross-validation valid) data points.
    score_type : instance implementing BaseScoreType signature
        The score to improve by adding one submission to the ensemble.
    ground_truths : instance of Predictions
        The ground truth.
    best_index_list : list of integers
        Indices of the current best model.

    Returns
    -------
    best_index_list : list of integers
        Indices of the models in the new combination. If the same as input,
        no models were found improving the score.
    """
    Predictions = type(ground_truths)
    best_predictions = combine_predictions(
        Predictions, predictions_list, best_index_list)
    best_score = score_type.score_function(ground_truths, best_predictions)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a
    # model is added several times, it's upweighted, leading to
    # integer-weighted ensembles.
    r = np.arange(len(predictions_list))
    # Randomization doesn't matter, only in case of exact equality.
    # np.random.shuffle(r)
    # print r
    for i in r:
        # try to append the ith prediction to the current best predictions
        new_index_list = np.append(best_index_list, i)
        combined_predictions = combine_predictions(
            Predictions, predictions_list, new_index_list)
        new_score = score_type.score_function(
            ground_truths, combined_predictions)
        if score_type.is_lower_the_better:
            is_improved = new_score < best_score - min_improvement
        else:
            is_improved = new_score > best_score + min_improvement
        if is_improved:
            best_index = i
            best_score = new_score
    if best_index > -1:
        return np.append(best_index_list, best_index), best_score
    else:
        return best_index_list, best_score


def blend_on_fold(predictions_list, ground_truths_valid, score_type,
                  max_n_ensemble=80, min_improvement=0.0):
    """Construct the best model combination on a single fold.

    Using greedy forward selection with replacement. See
    http://www.cs.cornell.edu/~caruana/ctp/ct.papers/
    caruana.icml04.icdm06long.pdf.
    Then sets foldwise contributivity.

    Parameters
    ----------
    force_ensemble : boolean
        To force include deleted models
    """
    # The submissions must have is_to_ensemble set to True. It is for
    # fogetting models. Users can also delete models in which case
    # we make is_valid false. We then only use these models if
    # force_ensemble is True.
    # We can further bag here which should be handled in config (or
    # ramp table.) Or we could bag in get_next_best_single_fold
    if len(predictions_list) == 0:
        return None, None, None, None
    valid_scores = [score_type.score_function(ground_truths_valid, predictions)
                    for predictions in predictions_list]
    if score_type.is_lower_the_better:
        best_prediction_index = np.argmin(valid_scores)
    else:
        best_prediction_index = np.argmax(valid_scores)
    score = valid_scores[best_prediction_index]
    best_index_list = np.array([best_prediction_index])
    is_improved = True
    while is_improved and len(best_index_list) < max_n_ensemble:
        print('\t{}: {}'.format(best_index_list, score))
        old_best_index_list = best_index_list
        best_index_list, score = _get_next_best_submission(
            predictions_list, ground_truths_valid, score_type, best_index_list,
            min_improvement)
        is_improved = len(best_index_list) != len(old_best_index_list)
    return best_index_list
    # we share a unit of 1. among the contributive submissions
    # unit_contributivity = 1. / len(best_index_list)
    # for i in best_index_list:
    #     selected_submissions_on_fold[i].contributivity +=\
    #         unit_contributivity
    # combined_predictions = combine_predictions_list(
    #     predictions_list, index_list=best_index_list)
    # best_predictions = predictions_list[best_index_list[0]]

    # test_predictions_list = [
    #     submission_on_fold.test_predictions
    #     for submission_on_fold in selected_submissions_on_fold
    # ]
    # if any(test_predictions_list) is None:
    #     logger.error("Can't compute combined test score," +
    #                  " some submissions are untested.")
    #     combined_test_predictions = None
    #     best_test_predictions = None
    # else:
    #     combined_test_predictions = combine_predictions_list(
    #         test_predictions_list, index_list=best_index_list)
    #     best_test_predictions = test_predictions_list[best_index_list[0]]

    # return combined_predictions, best_predictions,\
    #     combined_test_predictions, best_test_predictions
