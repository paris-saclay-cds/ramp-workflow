# coding: utf-8
"""
Scoring utilities
"""
import numpy as np
import pandas as pd

from .pretty_print import IS_COLOR_TERM
from .pretty_print import print_warning


def reorder_df_scores(df_scores, score_types):
    """Reorder scores according to the order in score_types.

    Parameters
    ----------
    df_scores : pd.DataFrame
        the score dataframe
    score_types : list of score types

    Returns
    -------
    df_scores : the dataframe with reordered scores
    """
    try:
        # try to re-order columns/rows in the printed array
        # we may not have all train, valid, test, so need to select
        index_order = np.array(['train', 'valid', 'test'])
        ordered_index = index_order[np.isin(index_order, df_scores.index)]
        df_scores = df_scores.loc[
            ordered_index, [score_type.name for score_type in score_types]]
    except Exception:
        print_warning("Couldn't re-order the score matrix..")
    return df_scores


def mean_score_matrix(df_scores_list, score_types):
    u"""Construct a mean ± std score dataframe from a list of score dataframes.

    Parameters
    ----------
    df_scores_list : list of pd.DataFrame
        a list of score data frames to average
    score_types : list of score types
        a list of score types to use

    Returns
    -------
    df_scores : the mean ± std score dataframe

    """
    scores = np.array([df_scores.values for df_scores in df_scores_list])
    meanss = scores.mean(axis=0)
    stdss = scores.std(axis=0)
    precisions = [st.precision for st in score_types]
    precisions.append(1)  # for time
    # we use unicode no break space so split in print_df_scores works
    if IS_COLOR_TERM:
        strs = np.array([[
            u'{val}\u00A0±\u00A0{std}'.format(
                val=round(mean, prec),
                std=round(std, prec + 1))
            for mean, std, prec in zip(means, stds, precisions)]
            for means, stds in zip(meanss, stdss)])
    else:
        strs = np.array([[
            u'{val} +- {std}'.format(
                val=round(mean, prec),
                std=round(std, prec + 1))
            for mean, std, prec in zip(means, stds, precisions)]
            for means, stds in zip(meanss, stdss)])
    df_scores = pd.DataFrame(
        strs, columns=df_scores_list[0].columns, index=df_scores_list[0].index)
    return df_scores


def score_matrix_from_scores(score_types, steps, scoress):
    """Construct a score dataframe from a matrix of scores.

    Parameters
    ----------
    score_types : list of score types
        a list of score types to use, score_type.name serves as column index
    steps : a list of strings
        subset of ['train', 'valid', 'test'], serves as row index

    Returns
    -------
    df_scores : the score dataframe
    """
    results = []
    for step, scores in zip(steps, scoress):
        for score_type, score in zip(score_types, scores):
            results.append(
                {'step': str(step), 'score': score_type.name, 'value': score})
    df_scores = pd.DataFrame(results)
    df_scores = df_scores.set_index(['step', 'score'])['value']
    df_scores = df_scores.unstack()
    df_scores = reorder_df_scores(df_scores, score_types)
    return df_scores


def score_matrix(score_types, ground_truth, predictions):
    """Construct a score dataframe by scoring predictions against ground truth.

    Parameters
    ----------
    score_types : list of score types
        a list of score types to use, score_type.name serves as column index
    ground_truth : dict of Predictions
        the ground truth data
    predictions : dict of Predictions
        the predicted data

    Returns
    -------
    df_scores : pd.DataFrame
        table of scores (rows = train/valid/test steps, columns = scores)
    """
    if set(ground_truth.keys()) != set(predictions.keys()):
        raise ValueError(('Predictions and ground truth steps '
                          'do not match:\n'
                          ' * predictions = {} \n'
                          ' * ground_truth = {} ')
                         .format(set(predictions.keys()),
                                 set(ground_truth.keys())))
    steps = ground_truth.keys()
    scoress = [[
        score_type.score_function(ground_truth[step], predictions[step])
        for score_type in score_types] for step in ground_truth]
    return score_matrix_from_scores(score_types, steps, scoress)


def round_df_scores(df_scores, score_types):
    """Round scores to the precision set in the score type.

    Parameters
    ----------
    df_scores : pd.DataFrame
        the score dataframe
    score_types : list of score types

    Returns
    -------
    df_scores : the dataframe with rounded scores
    """
    df_scores_copy = df_scores.copy()
    for column, score_type in zip(df_scores_copy, score_types):
        df_scores_copy[column] = [round(score, score_type.precision)
                                  for score in df_scores_copy[column]]
    return df_scores_copy
