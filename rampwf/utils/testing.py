# coding: utf-8

"""Provide utils to test ramp-kits."""
import os
import shutil
import sys

import numpy as np
import pandas as pd

from .combine import blend_on_fold
from .io import load_y_pred
from .importing import import_module_from_source
from .pretty_print import print_title, print_df_scores
from .notebook import execute_notebook, convert_notebook
from .scoring import round_df_scores, mean_score_matrix
from .submission import (bag_submissions, run_submission_on_cv_fold,
                         run_submission_on_full_train)


def assert_notebook(ramp_kit_dir='.'):
    print('----------------------------')
    convert_notebook(ramp_kit_dir)
    execute_notebook(ramp_kit_dir)


def assert_read_problem(ramp_kit_dir='.'):
    # allowing external imports from the external_imports folder if it exists
    ext_imp_dir = os.path.join(ramp_kit_dir, "external_imports")
    if os.path.exists(ext_imp_dir) and ext_imp_dir not in sys.path:
        sys.path.append(ext_imp_dir)
    # giving a random name to the module so it passes looped tests
    return import_module_from_source(
        os.path.join(ramp_kit_dir, 'problem.py'), 'problem'
    )


def assert_title(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Testing {}'.format(problem.problem_title))


def assert_data(ramp_kit_dir='.', ramp_data_dir='.', data_label=None):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Reading train and test files from {}/data/{} ...'.format(
        ramp_data_dir, f'{data_label}/' if data_label is not None else ""))
    kwargs = {}
    if data_label is not None:
        kwargs['data_label'] = data_label
    X_train, y_train = problem.get_train_data(path=ramp_data_dir, **kwargs)
    X_test, y_test = problem.get_test_data(path=ramp_data_dir, **kwargs)
    return X_train, y_train, X_test, y_test


def assert_cv(ramp_kit_dir='.', ramp_data_dir='.', data_label=None):
    problem = assert_read_problem(ramp_kit_dir)
    kwargs = {}
    if data_label is not None:
        kwargs['data_label'] = data_label
    X_train, y_train = problem.get_train_data(path=ramp_data_dir, **kwargs)
    print_title('Reading cv ...')
    cv = list(problem.get_cv(X_train, y_train))
    return cv


def assert_score_types(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    score_types = problem.score_types
    return score_types


def assert_submission(ramp_kit_dir='.', ramp_data_dir='.',
                      ramp_submission_dir='submissions', data_label=None,
                      submission='starting_kit', is_pickle=False,
                      is_partial_train=False, save_output=False,
                      retrain=False):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be tested for submission.
    ramp_data_dir : str, default='.'
        The directory of the data.
    ramp_submission_dir : str, default='./submissions'
        The directory of the submissions.
    data_label : str, default=None
        The subdirectory of data in /data and training outputs in
        /submissions/<submission>/training_output
    submission : str, default='starting_kit'
        The name of the submission to be tested.
    is_pickle : bool, default is False
        Whether to pickle the trained workflow or not.
    is_partial_train : bool, default is False
        Whether to partial train a trained workflow, pickled before.
        workflow.train_submission needs to accept prev_trained_workflow.
    save_y_preds : bool, default is False
        Whether to store the predictions.
    retrain : bool, default is False
        Whether to train the workflow on the full training set and
        test on the test set.
    """
    problem = assert_read_problem(ramp_kit_dir)
    assert_title(ramp_kit_dir)
    X_train, y_train, X_test, y_test = assert_data(
        ramp_kit_dir, ramp_data_dir, data_label)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir, data_label)
    score_types = assert_score_types(ramp_kit_dir)

    # module_path = os.path.join(ramp_kit_dir, 'submissions', submission)
    submission_path = os.path.join(ramp_submission_dir, submission)
    print_title('Training {} ...'.format(submission_path))

    training_output_path = ''
    if is_pickle or save_output:
        # creating <submission_path>/<submission>/training_output dir
        # optionally
        # <submission_path>/<submission>/training_output/<data_label>
        training_output_path = os.path.join(
            submission_path, 'training_output')
        if data_label is not None:
            training_output_path = os.path.join(
                training_output_path, data_label)
        if not os.path.exists(training_output_path):
            os.makedirs(training_output_path)

        print('Training output path: {}'.format(training_output_path))

    # saving predictions for CV bagging after the CV loop
    predictions_valid_list = []
    predictions_test_list = []
    df_scores_list = []

    for fold_i, fold in enumerate(cv):
        fold_output_path = ''
        if is_pickle or save_output:
            # creating <submission_path>/<submission>/training_output/fold_<i>
            fold_output_path = os.path.join(
                training_output_path, 'fold_{}'.format(fold_i))
            if not os.path.exists(fold_output_path):
                os.makedirs(fold_output_path)
        print_title('CV fold {}'.format(fold_i))

        predictions_valid, predictions_test, df_scores = \
            run_submission_on_cv_fold(
                problem, submission_path, fold, X_train, y_train,
                X_test, y_test, is_pickle, is_partial_train, save_output,
                fold_output_path, ramp_data_dir)
        if save_output:
            filename = os.path.join(fold_output_path, 'scores.csv')
            df_scores.to_csv(filename)
        df_scores_rounded = round_df_scores(df_scores, score_types)
        print_df_scores(df_scores_rounded, indent='\t')

        # saving predictions for CV bagging after the CV loop
        df_scores_list.append(df_scores)
        predictions_valid_list.append(predictions_valid)
        predictions_test_list.append(predictions_test)

    print_title('----------------------------')
    print_title('Mean CV scores')
    print_title('----------------------------')
    df_mean_scores = mean_score_matrix(df_scores_list, score_types)
    print_df_scores(df_mean_scores, indent='\t')

    if retrain:
        # We retrain on the full training set
        print_title('----------------------------')
        print_title('Retrain scores')
        print_title('----------------------------')
        run_submission_on_full_train(
            problem, submission_path, X_train, y_train, X_test, y_test,
            score_types, is_pickle, save_output, training_output_path,
            ramp_data_dir)
    bag_submissions(
        problem, cv, y_train, y_test, predictions_valid_list,
        predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=None,
        save_output=save_output)


def blend_submissions(submissions, ramp_kit_dir='.', ramp_data_dir='.',
                      ramp_submission_dir='.', data_label=None,
                      save_output=False, min_improvement=0.0,
                      score_type_index=0):
    """Blending submissions in a ramp-kit and compute contributivities.

    If save_output is True, we create three files:
    <ramp_submission_dir>/training_output/contributivities.csv
    <ramp_submission_dir>/training_output/bagged_scores_combined.csv
    <ramp_submission_dir>/training_output/bagged_scores_foldwise_best.csv

    Parameters
    ----------
    submissions : list of str
        List of submission names (folders in <ramp_submission_dir>).
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be blended.
    ramp_data_dir : str, default='.'
        The directory of the data.
    data_label : str, default=None
        The subdirectory of data in {ramp_data_dir}/data and training outputs
        in {ramp_kit_dir}/submissions/<submission>/training_output
    ramp_submission_dir : str, default='./submissions'
        The directory of the submissions.
    save_output : bool, default is False
        Whether to store the blending results.
    min_improvement : float, default is 0.0
        The minimum improvement under which greedy blender is stopped.
    """
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Blending {}'.format(problem.problem_title))
    X_train, y_train, X_test, y_test = assert_data(
        ramp_kit_dir, ramp_data_dir, data_label)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir, data_label)
    valid_is_list = [valid_is for (train_is, valid_is) in cv]
    score_types = assert_score_types(ramp_kit_dir)
    n_folds = len(valid_is_list)
    contributivitys = np.zeros((len(submissions), n_folds))

    combined_predictions_valid_list = []
    foldwise_best_predictions_valid_list = []
    combined_predictions_test_list = []
    foldwise_best_predictions_test_list = []
    for fold_i, valid_is in enumerate(valid_is_list):
        print_title('CV fold {}'.format(fold_i))
        ground_truths_valid = problem.Predictions(
            y_true=y_train, fold_is=valid_is)
        predictions_valid_list = []
        predictions_test_list = []
        for submission in submissions:
            module_path = os.path.join(ramp_submission_dir, submission)
            training_output_path = os.path.join(
                module_path, 'training_output')
            if data_label is not None:
                training_output_path = os.path.join(
                    training_output_path, data_label)
            fold_output_path = os.path.join(
                training_output_path, 'fold_{}'.format(fold_i))
            y_pred_train = load_y_pred(
                problem, data_path=ramp_data_dir,
                input_path=fold_output_path, suffix='train')
            y_pred_test = load_y_pred(
                problem, data_path=ramp_data_dir,
                input_path=fold_output_path, suffix='test')
            predictions_valid = problem.Predictions(
                y_pred=y_pred_train, fold_is=valid_is)
            predictions_valid_list.append(predictions_valid)
            predictions_test = problem.Predictions(y_pred=y_pred_test)
            predictions_test_list.append(predictions_test)

        best_index_list = blend_on_fold(
            predictions_valid_list, ground_truths_valid,
            score_types[score_type_index],
            min_improvement=min_improvement)

        # we share a unit of 1. among the contributive submissions
        unit_contributivity = 1. / len(best_index_list)
        for i in best_index_list:
            contributivitys[i, fold_i] += unit_contributivity

        combined_predictions_valid_list.append(
            problem.Predictions.combine(
                predictions_valid_list, best_index_list))
        foldwise_best_predictions_valid_list.append(
            predictions_valid_list[best_index_list[0]])
        combined_predictions_test_list.append(
            problem.Predictions.combine(
                predictions_test_list, best_index_list))
        foldwise_best_predictions_test_list.append(
            predictions_test_list[best_index_list[0]])

    contributivitys /= n_folds
    contributivitys_df = pd.DataFrame()
    contributivitys_df['submission'] = np.array(submissions)
    contributivitys_df['contributivity'] = np.zeros(len(submissions))
    for fold_i in range(n_folds):
        c_i = contributivitys[:, fold_i]
        contributivitys_df['fold_{}'.format(fold_i)] = c_i
        contributivitys_df['contributivity'] += c_i
    percentage_factor = 100 / contributivitys_df['contributivity'].sum()
    contributivitys_df['contributivity'] *= percentage_factor
    rounded = contributivitys_df['contributivity'].round().astype(int)
    contributivitys_df['contributivity'] = rounded
    contributivitys_df = contributivitys_df.sort_values(
        'contributivity', ascending=False)
    print(contributivitys_df.to_string(index=False))

    if save_output:
        training_output_path = os.path.join(
            ramp_submission_dir, 'training_output')
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)
        if data_label is not None:
            training_output_path = os.path.join(
                training_output_path, data_label)
            if not os.path.exists(training_output_path):
                os.makedirs(training_output_path)
        contributivitys_df.to_csv(os.path.join(
            training_output_path, 'contributivities.csv'), index=False)

    # bagging the foldwise ensembles
    bag_submissions(
        problem, cv, y_train, y_test, combined_predictions_valid_list,
        combined_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=score_type_index,
        save_output=save_output, score_table_title='Combined bagged scores',
        score_f_name_prefix='combined')
    if save_output:
        shutil.move(
            os.path.join(training_output_path, 'bagged_scores.csv'),
            os.path.join(training_output_path, 'bagged_scores_combined.csv'))

    # bagging the foldwise best submissions
    bag_submissions(
        problem, cv, y_train, y_test, foldwise_best_predictions_valid_list,
        foldwise_best_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=score_type_index,
        save_output=save_output,
        score_table_title='Foldwise best bagged scores',
        score_f_name_prefix='foldwise_best')
    if save_output:
        shutil.move(
            os.path.join(training_output_path, 'bagged_scores.csv'),
            os.path.join(
                training_output_path, 'bagged_scores_foldwise_best.csv'))
