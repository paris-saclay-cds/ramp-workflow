# coding: utf-8

"""Provide utils to test ramp-kits."""
from __future__ import print_function

import os
import imp
from subprocess import call
from os.path import join, abspath

import numpy as np
import cloudpickle as pickle
from .combine import get_score_cv_bags


def _delete_line_from_file(f_name, line_to_delete):
    with open(f_name, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line != line_to_delete:
                f.write(line)
        f.truncate()


def execute_notebook(ramp_kit_dir='.'):
    problem_name = abspath(ramp_kit_dir).split('/')[-1]
    print('Testing if the notebook can be executed')
    call(
        'jupyter nbconvert --execute {}/{}_starting_kit.ipynb '.format(
            ramp_kit_dir, problem_name) +
        '--ExecutePreprocessor.kernel_name=$IPYTHON_KERNEL ' +
        '--ExecutePreprocessor.timeout=600', shell=True)


def convert_notebook(ramp_kit_dir='.'):
    problem_name = abspath(ramp_kit_dir).split('/')[-1]
    print('Testing if the notebook can be converted to html')
    call('jupyter nbconvert --to html {}/{}_starting_kit.ipynb'.format(
        ramp_kit_dir, problem_name), shell=True)
    _delete_line_from_file(
        '{}/{}_starting_kit.html'.format(ramp_kit_dir, problem_name),
        '<link rel="stylesheet" href="custom.css">\n')


def assert_notebook(ramp_kit_dir='.'):
    print('----------------------------')
    convert_notebook(ramp_kit_dir)
    execute_notebook(ramp_kit_dir)


def assert_read_problem(ramp_kit_dir='.'):
    problem = imp.load_source('', join(ramp_kit_dir, 'problem.py'))
    return problem


def assert_title(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    print('Testing {}'.format(problem.problem_title))


def assert_data(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    print('Reading train and test files from {}/data ...'.format(
        ramp_data_dir))
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    X_test, y_test = problem.get_test_data(path=ramp_data_dir)
    return X_train, y_train, X_test, y_test


def assert_cv(ramp_kit_dir='.', ramp_data_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    print('Reading cv ...')
    cv = list(problem.get_cv(X_train, y_train))
    return cv


def assert_score_types(ramp_kit_dir='.'):
    problem = assert_read_problem(ramp_kit_dir)
    score_types = problem.score_types
    return score_types


def _print_cv_scores(scores, score_types, step):
    means = scores.mean(axis=0)
    stds = scores.std(axis=0)
    for mean, std, score_type in zip(means, stds, score_types):
        # If std is a NaN
        if std != std:
            result = '{step} {name} = {val}'.format(
                step=step, name=score_type.name, val=mean)
        else:
            result = '{step} {name} = {val} ± {std}'.format(
                step=step,
                name=score_type.name,
                val=round(mean, score_type.precision),
                std=round(std, score_type.precision))
        print(result)


def _print_single_score(score_type, ground_truth, predictions, step,
                        indent=''):
    score = score_type.score_function(ground_truth, predictions)
    rounded_score = round(score, score_type.precision)
    print('{indent}{step} {name} = {val}'.format(
        indent=indent, step=step, name=score_type.name, val=rounded_score))
    return score


def _save_y_pred(problem, y_pred, data_path='.', output_path='.',
                 suffix='test'):
    try:
        # We try using custom made problem.save_y_pred
        # it may need to re-read the data, e.g., for ids, so we send
        # it the data path
        problem.save_y_pred(y_pred, data_path, output_path, suffix)
    except AttributeError:
        # We fall back to numpy save
        try:
            y_pred_f_name = join(output_path, 'y_pred_{}.csv'.format(suffix))
            np.savetxt(y_pred_f_name, y_pred)
        except Exception as e:
            print("Warning: model can't be saved.")
            print(e)
            print('Consider implementing custom save_y_pred in problem.py')
            print('See https://github.com/ramp-kits/kaggle_seguro/blob/master/problem.py')  # noqa


def assert_submission(ramp_kit_dir='.', ramp_data_dir='.',
                      submission='starting_kit', is_pickle=False,
                      save_y_preds=False):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str, (default='.')
        The directory of the ramp-kit to be tested for submission.

    ramp_data_dir : str, (default='.')
        The directory of the data

    submission_name : str, (default='starting_kit')
        The name of the submission to be tested.

    Returns
    -------
    None

    """
    problem = assert_read_problem(ramp_kit_dir)
    assert_title(ramp_kit_dir)
    X_train, y_train, X_test, y_test = assert_data(ramp_kit_dir, ramp_data_dir)
    cv = assert_cv(ramp_kit_dir, ramp_data_dir)
    score_types = assert_score_types(ramp_kit_dir)

    module_path = join(ramp_kit_dir, 'submissions', submission)
    print('Training {} ...'.format(module_path))

    if is_pickle or save_y_preds:
        # creating submissions/<submission>/training_output dir
        training_output_path = join(module_path, 'training_output')
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)

    # saving scores for mean/std stats after the CV loop
    train_train_scoress = np.empty((len(cv), len(score_types)))
    train_valid_scoress = np.empty((len(cv), len(score_types)))
    test_scoress = np.empty((len(cv), len(score_types)))

    # saving predictions for CV bagging after the CV loop
    predictions_train_valid_list = []
    predictions_test_list = []

    for fold_i, (train_is, valid_is) in enumerate(cv):
        if is_pickle or save_y_preds:
            # creating submissions/<submission>/training_output/fold_<i> dir
            fold_output_path = join(
                training_output_path, 'fold_{}'.format(fold_i))
            if not os.path.exists(fold_output_path):
                os.mkdir(fold_output_path)

        trained_workflow = problem.workflow.train_submission(
            module_path, X_train, y_train, train_is=train_is)

        if is_pickle:
            try:
                model_file = join(fold_output_path, 'model.pkl')
                with open(model_file, 'wb') as pickle_file:
                    pickle.dump(trained_workflow, pickle_file)
                with open(model_file, 'r') as pickle_file:
                    trained_workflow = pickle.load(pickle_file)
            except Exception as e:
                print("Warning: model can't be pickled.")
                print(e)

        y_pred_train = problem.workflow.test_submission(
            trained_workflow, X_train)
        predictions_train_train = problem.Predictions(
            y_pred=y_pred_train[train_is])
        ground_truth_train_train = problem.Predictions(
            y_true=y_train[train_is])
        predictions_train_valid = problem.Predictions(
            y_pred=y_pred_train[valid_is])
        ground_truth_train_valid = problem.Predictions(
            y_true=y_train[valid_is])

        y_pred_test = problem.workflow.test_submission(
            trained_workflow, X_test)
        predictions_test = problem.Predictions(y_pred=y_pred_test)
        ground_truth_test = problem.Predictions(y_true=y_test)

        # saving predictions for CV bagging after the CV loop
        predictions_train_valid_list.append(predictions_train_valid)
        predictions_test_list.append(predictions_test)

        if save_y_preds:
            _save_y_pred(
                problem, y_pred_train, data_path=ramp_data_dir,
                output_path=fold_output_path, suffix='train')
            _save_y_pred(
                problem, y_pred_test, data_path=ramp_data_dir,
                output_path=fold_output_path, suffix='test')

        print('CV fold {}'.format(fold_i))
        for score_type_i, score_type in enumerate(score_types):
            train_train_scoress[fold_i, score_type_i] = _print_single_score(
                score_type, ground_truth_train_train, predictions_train_train,
                step='train', indent='\t')
            train_valid_scoress[fold_i, score_type_i] = _print_single_score(
                score_type, ground_truth_train_valid, predictions_train_valid,
                step='valid', indent='\t')
            test_scoress[fold_i, score_type_i] = _print_single_score(
                score_type, ground_truth_test, predictions_test,
                step='test', indent='\t')

    print('----------------------------')
    print('Mean CV scores')
    print('----------------------------')
    _print_cv_scores(train_train_scoress, score_types, step='train')
    _print_cv_scores(train_valid_scoress, score_types, step='valid')
    _print_cv_scores(test_scoress, score_types, step='test')

    # We retrain on the full training set
    print('----------------------------')
    print('Retrain scores')
    print('----------------------------')
    trained_workflow = problem.workflow.train_submission(
        module_path, X_train, y_train)
    y_pred_train = problem.workflow.test_submission(trained_workflow, X_train)
    predictions_train = problem.Predictions(y_pred=y_pred_train)
    ground_truth_train = problem.Predictions(y_true=y_train)
    y_pred_test = problem.workflow.test_submission(trained_workflow, X_test)
    predictions_test = problem.Predictions(y_pred=y_pred_test)
    ground_truth_test = problem.Predictions(y_true=y_test)
    for score_type in score_types:
        _print_single_score(
            score_type, ground_truth_train, predictions_train, step='train')
        _print_single_score(
            score_type, ground_truth_test, predictions_test, step='test')
    if save_y_preds:
        _save_y_pred(
            problem, y_pred_train, data_path=ramp_data_dir,
            output_path=training_output_path, suffix='retrain_train')
        _save_y_pred(
            problem, y_pred_test, data_path=ramp_data_dir,
            output_path=training_output_path, suffix='retrain_test')

    print('----------------------------')
    print('Bagged scores')
    print('----------------------------')
    valid_is_list = [valid_is for (train_is, valid_is) in cv]
    ground_truths_train = problem.Predictions(y_true=y_train)
    ground_truths_test = problem.Predictions(y_true=y_test)
    score_type = score_types[0]
    bagged_train_valid_predictions, bagged_train_valid_scores =\
        get_score_cv_bags(
            problem.Predictions, score_type, predictions_train_valid_list,
            ground_truths_train, test_is_list=valid_is_list)
    bagged_test_predictions, bagged_test_scores = get_score_cv_bags(
        problem.Predictions, score_type, predictions_test_list,
        ground_truths_test)
    print('valid {} = {}'.format(
        score_type.name, round(
            bagged_train_valid_scores[-1], score_type.precision)))
    print('test {} = {}'.format(
        score_type.name, round(bagged_test_scores[-1], score_type.precision)))
    if save_y_preds:
        # y_pred_bagged_train.csv contains _out of sample_ (validation)
        # predictions, but not for all points (contains nans)
        _save_y_pred(
            problem, bagged_train_valid_predictions.y_pred,
            data_path=ramp_data_dir, output_path=training_output_path,
            suffix='bagged_train')
        _save_y_pred(
            problem, bagged_test_predictions.y_pred, data_path=ramp_data_dir,
            output_path=training_output_path, suffix='bagged_test')
