# coding: utf-8

"""The :mod:`rampwf.utils.testing` submodule provide utils to test ramp-kits"""
from __future__ import print_function

import imp
from os.path import join

import numpy as np


def assert_submission(ramp_kit_dir='./'):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str,
        The directory of the ramp-kit to be tested for submission.

    Returns
    -------
    None

    """
    problem = imp.load_source('', join(ramp_kit_dir + 'problem.py'))
    print('Testing {}'.format(problem.problem_title))
    print('Reading file ...')
    X, y = problem.get_data(path=ramp_kit_dir)
    prediction_labels = problem.prediction_labels
    score_types = problem.score_types
    print('Training model ...')
    cv = list(problem.get_cv(X, y))
    module_path = join(ramp_kit_dir, 'submissions/starting_kit')
    scoress = np.empty((len(cv), len(score_types)))
    for fold_i, (train_is, test_is) in enumerate(cv):
        trained_workflow = problem.workflow.train_submission(
            module_path, X, y, train_is=train_is)
        y_pred = problem.workflow.test_submission(trained_workflow, X)
        predictions = problem.prediction_type.Predictions(
            y_pred=y_pred[test_is], labels=prediction_labels)
        ground_truth = problem.prediction_type.Predictions(
            y_true=y[test_is], labels=prediction_labels)
    print('CV fold {}'.format(fold_i))
    for score_type_i, score_type in enumerate(score_types):
        score = score_type.score_function(ground_truth, predictions)
        scoress[fold_i, score_type_i] = score
        print('\t{} = {}'.format(
            score_type.name, round(score, score_type.precision)))
    print('----------------------------')
    means = scoress.mean(axis=0)
    stds = scoress.std(axis=0)
    for mean, std, score_type in zip(means, stds, score_types):
        print('{} = {} ± {}'.format(
            score_type.name, round(mean, score_type.precision),
            round(std, score_type.precision + 1)))


def assert_backend(ramp_kit_dir='./'):
    """Helper to test the backend from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str,
        The directory of the ramp-kit to be tested for submission.

    Returns
    -------
    None

    """
    problem = imp.load_source('', ramp_kit_dir + '/backend.py')
    print('Preparing {} data'.format(problem.problem_title))
    problem.prepare_data()
    print('Testing {} backend'.format(problem.problem_title))
    print('Reading file ...')
    X_train, y_train = problem.get_train_data(path=ramp_kit_dir)
    X_test, y_test = problem.get_test_data(path=ramp_kit_dir)
    prediction_labels = problem.prediction_labels
    score_types = problem.score_types
    print('Training model ...')
    cv = list(problem.get_cv(X_train, y_train))
    module_path = ramp_kit_dir + '/submissions/starting_kit'
    train_train_scoress = np.empty((len(cv), len(score_types)))
    train_valid_scoress = np.empty((len(cv), len(score_types)))
    test_scoress = np.empty((len(cv), len(score_types)))
    for fold_i, (train_is, valid_is) in enumerate(cv):
        trained_workflow = problem.workflow.train_submission(
            module_path, X_train, y_train, train_is=train_is)

        y_pred_train = problem.workflow.test_submission(trained_workflow,
                                                        X_train)
        predictions_train_train = problem.prediction_type.Predictions(
            y_pred=y_pred_train[train_is], labels=prediction_labels)
        ground_truth_train_train = problem.prediction_type.Predictions(
            y_true=y_train[train_is], labels=prediction_labels)
        predictions_train_valid = problem.prediction_type.Predictions(
            y_pred=y_pred_train[valid_is], labels=prediction_labels)
        ground_truth_train_valid = problem.prediction_type.Predictions(
            y_true=y_train[valid_is], labels=prediction_labels)

        y_pred_test = problem.workflow.test_submission(trained_workflow,
                                                       X_test)
        predictions_test = problem.prediction_type.Predictions(
            y_pred=y_pred_test, labels=prediction_labels)
        ground_truth_test = problem.prediction_type.Predictions(
            y_true=y_test, labels=prediction_labels)

        print('CV fold {}'.format(fold_i))
        for score_type_i, score_type in enumerate(score_types):
            score = score_type.score_function(
                ground_truth_train_train, predictions_train_train)
            train_train_scoress[fold_i, score_type_i] = score
            print('\ttrain {} = {}'.format(
                score_type.name, round(score, score_type.precision)))

            score = score_type.score_function(
                ground_truth_train_valid, predictions_train_valid)
            train_valid_scoress[fold_i, score_type_i] = score
            print('\tvalid {} = {}'.format(
                score_type.name, round(score, score_type.precision)))

            score = score_type.score_function(
                ground_truth_test, predictions_test)
            train_valid_scoress[fold_i, score_type_i] = score
            print('\ttest {} = {}'.format(
                score_type.name, round(score, score_type.precision)))

    print('----------------------------')
    means = train_train_scoress.mean(axis=0)
    stds = train_train_scoress.std(axis=0)
    for mean, std, score_type in zip(means, stds, score_types):
        print('train {} = {} ± {}'.format(
            score_type.name, round(mean, score_type.precision),
            round(std, score_type.precision + 1)))

    means = train_valid_scoress.mean(axis=0)
    stds = train_valid_scoress.std(axis=0)
    for mean, std, score_type in zip(means, stds, score_types):
        print('train {} = {} ± {}'.format(
            score_type.name, round(mean, score_type.precision),
            round(std, score_type.precision + 1)))

    means = test_scoress.mean(axis=0)
    stds = test_scoress.std(axis=0)
    for mean, std, score_type in zip(means, stds, score_types):
        print('train {} = {} ± {}'.format(
            score_type.name, round(mean, score_type.precision),
            round(std, score_type.precision + 1)))
