# coding: utf-8

"""Provide utils to test ramp-kits."""
from __future__ import print_function

import os
import imp
from collections import OrderedDict

import numpy as np
import pandas as pd
import cloudpickle as pickle

from .combine import get_score_cv_bags, blend_on_fold
from .notebook import execute_notebook, convert_notebook
from .colors import print_title, print_warning, print_df_scores
from .score import (score_matrix, score_matrix_from_scores, round_df_scores,
                    mean_score_matrix)


def assert_notebook(ramp_kit_dir='.'):
    print('----------------------------')
    convert_notebook(ramp_kit_dir)
    execute_notebook(ramp_kit_dir)


def assert_read_problem(ramp_kit_dir='.'):
    # giving a random name to the module so it passes looped tests
    module_name = str(int(1000000000 * np.random.rand()))
    problem = imp.load_source(module_name,
                              os.path.join(ramp_kit_dir, 'problem.py'))
    return problem


def assert_title(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Testing {}'.format(problem.problem_title))


def assert_data(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Reading train and test files from {}/data ...'.format(
        ramp_data_dir))
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    X_test, y_test = problem.get_test_data(path=ramp_data_dir)
    return X_train, y_train, X_test, y_test


def assert_cv(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    print_title('Reading cv ...')
    cv = list(problem.get_cv(X_train, y_train))
    return cv


def assert_score_types(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    score_types = problem.score_types
    return score_types


def _save_y_pred(problem, y_pred, data_path='.', output_path='.',
                 suffix='test'):
    """Save a prediction vector in file.

    If problem.save_y_pred is implemented, y_pred is passed to it. Otherwise,
    np.savez_compressed is used on y_pred. If it crashes, a warning is raised.
    The file is (typically) in
    submissions/<submission_name>/training_output/y_pred_<suffix>.npz or
    submissions/<submission_name>/training_output/fold_<i>/y_pred_<suffix>.npz.

    Parameters
    ----------
    problem : a problem object
        loaded from problem.py, may implement save_y_pred
    y_pred : a prediction vector
        a vector of predictions to be saved
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    output_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.npz will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.npz, can be used in
        problem.save_y_pred to, e.g., save only test predictions
    """
    try:
        # We try using custom made problem.save_y_pred
        # obligatory to implement if np.savez_compressed doesn't work on y_pred
        problem.save_y_pred(y_pred, data_path, output_path, suffix)
    except AttributeError:
        # We fall back to numpy savez_compressed
        try:
            y_pred_f_name = os.path.join(output_path,
                                         'y_pred_{}'.format(suffix))
            np.savez_compressed(y_pred_f_name, y_pred=y_pred)
        except Exception as e:
            print_warning(
                "Warning: model can't be saved.\n{}\n".format(e) +
                'Consider implementing custom save_y_pred in problem.py\n')


def _load_y_pred(problem, data_path='.', input_path='.', suffix='test'):
    """Load a file into a prediction vector.

    If problem.load_y_pred is implemented, y_pred is loaded by it. Otherwise,
    np.load is used. If it crashes, the exception is raised. The input file
    is (typically) in
    submissions/<submission_name>/training_output/y_pred_<suffix>.npz or
    submissions/<submission_name>/training_output/fold_<i>/y_pred_<suffix>.npz.

    Parameters
    ----------
    problem : a problem object
        loaded from problem.py, may implement save_y_pred
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    input_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.npz will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.npz
    """
    try:
        # We try using custom made problem.load_y_pred
        # obligatory to implement if np.load doesn't work
        return problem.load_y_pred(data_path, input_path, suffix)
    except AttributeError:
        # We fall back to numpy load
        y_pred_f_name = os.path.join(input_path,
                                     'y_pred_{}.npz'.format(suffix))
        return np.load(y_pred_f_name)['y_pred']


def _save_submission(problem, y_pred, data_path='.', output_path='.',
                     suffix='test'):
    """Custom save a prediction vector in file to, e.g., submit to Kaggle.

    If problem.save_submission is implemented, y_pred is passed to it.
    Otherwise nothing happens (the exception is caught silently).

    Parameters
    ----------
    problem : a problem object
        loaded from problem.py, may implement save_y_pred
    y_pred : a prediction vector
        a vector of predictions to be saved
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    output_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.csv will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.csv, can be used in
        problem.save_y_pred to, e.g., save only test predictions
    """
    try:
        # We try using custom made problem.save_submission
        # it may need to re-read the data, e.g., for ids, so we send
        # it the data path. See, e.g.,
        # https://github.com/ramp-kits/kaggle_seguro/blob/master/problem.py
        problem.save_submission(y_pred, data_path, output_path, suffix)
    except AttributeError:
        pass


def _pickle_model(fold_output_path, trained_workflow, model_name='model.pkl'):
    """Pickle and reload trained workflow.

    If workflow can't be pickled, print warning and return origina' workflow.

    Parameters
    ----------
    fold_output_path : str
        the path into which the model will be pickled
    trained_workflow : a rampwf.workflow
        the workflow to be pickled
    model_name : str (default='model.pkl')
        the file name of the pickled workflow
    Returns
    -------
    trained_workflow : a rampwf.workflow
        either the input workflow or the pickled and reloaded workflow
    """
    try:
        model_file = os.path.join(fold_output_path, model_name)
        with open(model_file, 'wb') as pickle_file:
            pickle.dump(trained_workflow, pickle_file)
        with open(model_file, 'r') as pickle_file:
            trained_workflow = pickle.load(pickle_file)
    except Exception as e:
        print_warning("Warning: model can't be pickled.")
        print_warning(e)
    return trained_workflow


def _train_test_submission(problem, module_path, X_train, y_train, X_test,
                           is_pickle, output_path,
                           model_name='model.pkl', train_is=None):
    """Train and test submission, on cv fold if train_is not none.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    module_path : str
        the path of the submission, typically submissions/<submission_name>
    X_train : a list of training instances
        returned by problem.get_train_data
    y_train : a list of training ground truth
        returned by problem.get_train_data
    X_train : a list of testing instances
        returned by problem.get_test_data
    is_pickle : boolean
        True if the model should be pickled
    output_path : str
        the path into which the model will be pickled
    model_name : str (default='model.pkl')
        the file name of the pickled workflow
    train_is : a list of integers (default=None)
        training indices from the cross-validation fold, if None, train
        on full set
    Returns
    -------
    y_pred_train : a list of predictions
        on the training (train_train and train_valid) set
    y_pred_test : a list of predictions
        on the test set
    """
    trained_workflow = problem.workflow.train_submission(
        module_path, X_train, y_train, train_is=train_is)
    if is_pickle:
        trained_workflow = _pickle_model(
            output_path, trained_workflow, model_name)

    y_pred_train = problem.workflow.test_submission(
        trained_workflow, X_train)
    y_pred_test = problem.workflow.test_submission(
        trained_workflow, X_test)
    return y_pred_train, y_pred_test


def _run_submission_on_cv_fold(problem, module_path, X_train, y_train,
                               X_test, y_test, score_types,
                               is_pickle, save_y_preds, fold_output_path,
                               fold, ramp_data_dir):
    """Run submission, compute and return predictions and scores on cv.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    module_path : str
        the path of the submission, typically submissions/<submission_name>
    X_train : a list of training instances
        returned by problem.get_train_data
    y_train : a list of training ground truth
        returned by problem.get_train_data
    X_train : a list of testing instances
        returned by problem.get_test_data
    y_test : a list of testing ground truth
        returned by problem.get_test_data
    score_types : a list of score types
        problem.score_types
    is_pickle : boolean
        True if the model should be pickled
    save_y_preds : boolean
        True if predictions should be written in files
    fold_output_path : str
        the path into which the model will be pickled
    fold : pair of lists of integers
        (train_is, valid_is) generated by problem.get_cv
    ramp_data_dir : str
        the directory of the data
    Returns
    -------
    predictions_train_valid : instance of Predictions
        on the validation set
    predictions_test : instance of Predictions
        on the test set
    df_scores : pd.DataFrame
        table of scores (rows = train/valid/test steps, columns = scores)
    """
    train_is, valid_is = fold
    y_pred_train, y_pred_test = _train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        fold_output_path, train_is=train_is)
    predictions_train_train = problem.Predictions(
        y_pred=y_pred_train[train_is])
    ground_truth_train_train = problem.Predictions(
        y_true=y_train[train_is])
    predictions_train_valid = problem.Predictions(
        y_pred=y_pred_train[valid_is])
    ground_truth_train_valid = problem.Predictions(
        y_true=y_train[valid_is])
    predictions_test = problem.Predictions(y_pred=y_pred_test)
    ground_truth_test = problem.Predictions(y_true=y_test)

    if save_y_preds:
        _save_y_pred(
            problem, y_pred_train, data_path=ramp_data_dir,
            output_path=fold_output_path, suffix='train')
        _save_y_pred(
            problem, y_pred_test, data_path=ramp_data_dir,
            output_path=fold_output_path, suffix='test')

    df_scores = score_matrix(
        score_types,
        ground_truth=OrderedDict([('train', ground_truth_train_train),
                                  ('valid', ground_truth_train_valid),
                                  ('test', ground_truth_test)]),
        predictions=OrderedDict([('train', predictions_train_train),
                                 ('valid', predictions_train_valid),
                                 ('test', predictions_test)]),
    )
    return predictions_train_valid, predictions_test, df_scores


def _run_submission_on_full_train(problem, module_path, X_train, y_train,
                                  X_test, y_test, score_types,
                                  is_pickle, save_y_preds, output_path,
                                  ramp_data_dir):
    """Run submission, compute predictions, and print scores on full train.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    module_path : str
        the path of the submission, typically submissions/<submission_name>
    X_train : a list of training instances
        returned by problem.get_train_data
    y_train : a list of training ground truth
        returned by problem.get_train_data
    X_train : a list of testing instances
        returned by problem.get_test_data
    y_test : a list of testing ground truth
        returned by problem.get_test_data
    score_types : a list of score types
        problem.score_types
    is_pickle : boolean
        True if the model should be pickled
    save_y_preds : boolean
        True if predictions should be written in files
    output_path : str
        the path into which the model will be pickled
    ramp_data_dir : str
        the directory of the data
    """
    y_pred_train, y_pred_test = _train_test_submission(
        problem, module_path, X_train, y_train, X_test, is_pickle,
        output_path, model_name='retrained_model.pkl')
    predictions_train = problem.Predictions(y_pred=y_pred_train)
    ground_truth_train = problem.Predictions(y_true=y_train)
    predictions_test = problem.Predictions(y_pred=y_pred_test)
    ground_truth_test = problem.Predictions(y_true=y_test)

    df_scores = score_matrix(
        score_types,
        ground_truth=OrderedDict([('train', ground_truth_train),
                                  ('test', ground_truth_test)]),
        predictions=OrderedDict([('train', predictions_train),
                                 ('test', predictions_test)]),
    )
    df_scores_rounded = round_df_scores(df_scores, score_types)
    print_df_scores(df_scores_rounded, score_types, indent='\t')

    if save_y_preds:
        _save_submission(
            problem, y_pred_train, data_path=ramp_data_dir,
            output_path=output_path, suffix='retrain_train')
        _save_submission(
            problem, y_pred_test, data_path=ramp_data_dir,
            output_path=output_path, suffix='retrain_test')


def _bag_submissions(problem, cv, y_train, y_test, predictions_valid_list,
                     predictions_test_list, training_output_path,
                     ramp_data_dir='.', score_type_index=0,
                     save_y_preds=False, score_table_title='Bagged scores',
                     score_f_name_prefix=''):
    print_title('----------------------------')
    print_title(score_table_title)
    print_title('----------------------------')
    valid_is_list = [valid_is for (train_is, valid_is) in cv]
    ground_truths_train = problem.Predictions(y_true=y_train)
    ground_truths_test = problem.Predictions(y_true=y_test)
    score_type = problem.score_types[score_type_index]
    bagged_valid_predictions, bagged_valid_scores =\
        get_score_cv_bags(
            score_type, predictions_valid_list,
            ground_truths_train, test_is_list=valid_is_list)
    bagged_test_predictions, bagged_test_scores = get_score_cv_bags(
        score_type, predictions_test_list, ground_truths_test)

    df_scores = score_matrix_from_scores(
        [score_type], ['valid', 'test'],
        [[bagged_valid_scores[-1]], [bagged_test_scores[-1]]])
    df_scores_rounded = round_df_scores(df_scores, [score_type])
    print_df_scores(df_scores_rounded, [score_type], indent='\t')

    if save_y_preds:
        # y_pred_bagged_train.csv contains _out of sample_ (validation)
        # predictions, but not for all points (contains nans)
        _save_submission(
            problem, bagged_valid_predictions.y_pred,
            data_path=ramp_data_dir, output_path=training_output_path,
            suffix='{}_bagged_train'.format(score_f_name_prefix))
        _save_submission(
            problem, bagged_test_predictions.y_pred, data_path=ramp_data_dir,
            output_path=training_output_path,
            suffix='{}_bagged_test'.format(score_f_name_prefix))
        # also save the partial combined scores (CV bagging learning curves)
        bagged_train_valid_scores_f_name = os.path.join(
            training_output_path,
            '{}_bagged_valid_scores.csv'.format(score_f_name_prefix))
        np.savetxt(bagged_train_valid_scores_f_name, bagged_valid_scores)
        bagged_test_scores_f_name = os.path.join(
            training_output_path,
            '{}_bagged_test_scores.csv'.format(score_f_name_prefix))
        np.savetxt(bagged_test_scores_f_name, bagged_test_scores)


def assert_submission(ramp_kit_dir='.', ramp_data_dir='.',
                      submission='starting_kit', is_pickle=False,
                      save_y_preds=False, retrain=False):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str, (default='.')
        the directory of the ramp-kit to be tested for submission

    ramp_data_dir : str, (default='.')
        the directory of the data

    submission_name : str, (default='starting_kit')
        the name of the submission to be tested
    """
    problem = assert_read_problem(ramp_kit_dir)
    assert_title(ramp_kit_dir)
    X_train, y_train, X_test, y_test = assert_data(ramp_kit_dir, ramp_data_dir)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir)
    score_types = assert_score_types(ramp_kit_dir)

    module_path = os.path.join(ramp_kit_dir, 'submissions', submission)
    print_title('Training {} ...'.format(module_path))

    training_output_path = ''
    if is_pickle or save_y_preds:
        # creating submissions/<submission>/training_output dir
        training_output_path = os.path.join(module_path, 'training_output')
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)

    # saving predictions for CV bagging after the CV loop
    predictions_valid_list = []
    predictions_test_list = []
    df_scores_list = []

    for fold_i, fold in enumerate(cv):
        fold_output_path = ''
        if is_pickle or save_y_preds:
            # creating submissions/<submission>/training_output/fold_<i> dir
            fold_output_path = os.path.join(
                training_output_path, 'fold_{}'.format(fold_i))
            if not os.path.exists(fold_output_path):
                os.mkdir(fold_output_path)
        print_title('CV fold {}'.format(fold_i))

        predictions_valid, predictions_test, df_scores =\
            _run_submission_on_cv_fold(
                problem, module_path, X_train, y_train, X_test, y_test,
                score_types, is_pickle, save_y_preds, fold_output_path,
                fold, ramp_data_dir)
        df_scores_rounded = round_df_scores(df_scores, score_types)
        print_df_scores(df_scores_rounded, score_types, indent='\t')

        # saving predictions for CV bagging after the CV loop
        df_scores_list.append(df_scores)
        predictions_valid_list.append(predictions_valid)
        predictions_test_list.append(predictions_test)

    print_title('----------------------------')
    print_title('Mean CV scores')
    print_title('----------------------------')
    df_mean_scores = mean_score_matrix(df_scores_list, score_types)
    print_df_scores(df_mean_scores, score_types, indent='\t')

    if retrain:
        # We retrain on the full training set
        print_title('----------------------------')
        print_title('Retrain scores')
        print_title('----------------------------')
        _run_submission_on_full_train(
            problem, module_path, X_train, y_train, X_test, y_test,
            score_types, is_pickle, save_y_preds, training_output_path,
            ramp_data_dir)
    _bag_submissions(
        problem, cv, y_train, y_test, predictions_valid_list,
        predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_y_preds=save_y_preds)


def blend_submissions(submissions, ramp_kit_dir='.', ramp_data_dir='.',
                      save_y_preds=False, min_improvement=0.0):
    problem = assert_read_problem(ramp_kit_dir)
    print_title('Blending {}'.format(problem.problem_title))
    X_train, y_train, X_test, y_test = assert_data(ramp_kit_dir, ramp_data_dir)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir)
    valid_is_list = [valid_is for (train_is, valid_is) in cv]
    score_types = assert_score_types(ramp_kit_dir)
    contributivitys = np.zeros(len(submissions))

    combined_predictions_valid_list = []
    foldwise_best_predictions_valid_list = []
    combined_predictions_test_list = []
    foldwise_best_predictions_test_list = []
    for fold_i, valid_is in enumerate(valid_is_list):
        print_title('CV fold {}'.format(fold_i))
        ground_truths_valid = problem.Predictions(y_true=y_train[valid_is])
        predictions_valid_list = []
        predictions_test_list = []
        for submission in submissions:
            module_path = os.path.join(ramp_kit_dir, 'submissions', submission)
            training_output_path = os.path.join(module_path, 'training_output')
            fold_output_path = os.path.join(
                training_output_path, 'fold_{}'.format(fold_i))
            y_pred_train = _load_y_pred(
                problem, data_path=ramp_data_dir,
                input_path=fold_output_path, suffix='train')
            y_pred_test = _load_y_pred(
                problem, data_path=ramp_data_dir,
                input_path=fold_output_path, suffix='test')
            predictions_valid = problem.Predictions(
                y_pred=y_pred_train[valid_is])
            predictions_valid_list.append(predictions_valid)
            predictions_test = problem.Predictions(y_pred=y_pred_test)
            predictions_test_list.append(predictions_test)

        best_index_list = blend_on_fold(
            predictions_valid_list, ground_truths_valid, score_types[0],
            min_improvement=min_improvement)

        # we share a unit of 1. among the contributive submissions
        unit_contributivity = 1. / len(best_index_list)
        for i in best_index_list:
            contributivitys[i] += unit_contributivity

        combined_predictions_valid_list.append(
            problem.Predictions.combine(predictions_valid_list))
        foldwise_best_predictions_valid_list.append(predictions_valid_list[0])
        combined_predictions_test_list.append(
            problem.Predictions.combine(predictions_test_list))
        foldwise_best_predictions_test_list.append(predictions_test_list[0])

    contributivitys /= len(cv)
    contributivitys_df = pd.DataFrame()
    contributivitys_df['submission'] = np.array(submissions)
    contributivitys_df['contributivity'] = np.round(contributivitys, 3)
    contributivitys_df = contributivitys_df.reset_index()
    contributivitys_df = contributivitys_df.sort_values(
        'contributivity', ascending=False)
    print(contributivitys_df.to_string(index=False))

    training_output_path = os.path.join(ramp_kit_dir, 'training_output')
    if not os.path.exists(training_output_path):
        os.mkdir(training_output_path)
    # bagging the foldwise ensembles
    _bag_submissions(
        problem, cv, y_train, y_test, combined_predictions_valid_list,
        combined_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_y_preds=save_y_preds, score_table_title='Combined bagged scores',
        score_f_name_prefix='foldwise_best')
    # bagging the foldwise best submissions
    _bag_submissions(
        problem, cv, y_train, y_test, foldwise_best_predictions_valid_list,
        foldwise_best_predictions_test_list, training_output_path,
        ramp_data_dir=ramp_data_dir, score_type_index=0,
        save_y_preds=save_y_preds,
        score_table_title='Foldwise best bagged scores',
        score_f_name_prefix='combined')
