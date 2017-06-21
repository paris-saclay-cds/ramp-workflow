# coding: utf-8

"""The :mod:`rampwf.utils.testing` submodule provide utils to test ramp-kits"""
from __future__ import print_function

import imp
from os.path import join, abspath
from os import system

import numpy as np


def assert_submission(ramp_kit_dir='./', ramp_data_dir='./data',
                      submission_name='starting_kit'):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str, (default='./')
        The directory of the ramp-kit to be tested for submission.

    ramp_data_dir : str, (default='./')
        The directory of the data

    submission_name : str, (default='starting_kit')
        The name of the submission to be tested.

    Returns
    -------
    None

    """
    problem = imp.load_source('', join(ramp_kit_dir, 'problem.py'))
    print('Testing {}'.format(problem.problem_title))
    print('Reading train and test files from {}/data ...'.format(
        ramp_data_dir))
    X_train, y_train = problem.get_train_data(path=ramp_data_dir)
    X_test, y_test = problem.get_test_data(path=ramp_data_dir)
    prediction_labels = problem.prediction_labels
    score_types = problem.score_types
    print('Training {}/submissions/{} ...'.format(ramp_kit_dir,
                                                  submission_name))
    cv = list(problem.get_cv(X_train, y_train))
    module_path = join(ramp_kit_dir, 'submissions', submission_name)
    train_train_scoress = np.empty((len(cv), len(score_types)))
    train_valid_scoress = np.empty((len(cv), len(score_types)))
    test_scoress = np.empty((len(cv), len(score_types)))
    for fold_i, (train_idxs, valid_idx) in enumerate(cv):
        trained_workflow = problem.workflow.train_submission(
            module_path, X_train, y_train, train_idxs=train_idxs)

        y_pred_train = problem.workflow.test_submission(trained_workflow,
                                                        X_train)
        predictions_train_train = problem.prediction_type.Predictions(
            y_pred=y_pred_train[train_idxs], labels=prediction_labels)
        ground_truth_train_train = problem.prediction_type.Predictions(
            y_true=y_train[train_idxs], labels=prediction_labels)
        predictions_train_valid = problem.prediction_type.Predictions(
            y_pred=y_pred_train[valid_idx], labels=prediction_labels)
        ground_truth_train_valid = problem.prediction_type.Predictions(
            y_true=y_train[valid_idx], labels=prediction_labels)

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
            test_scoress[fold_i, score_type_i] = score
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
        print('valid {} = {} ± {}'.format(
            score_type.name, round(mean, score_type.precision),
            round(std, score_type.precision + 1)))

    means = test_scoress.mean(axis=0)
    stds = test_scoress.std(axis=0)
    for mean, std, score_type in zip(means, stds, score_types):
        print('test {} = {} ± {}'.format(
            score_type.name, round(mean, score_type.precision),
            round(std, score_type.precision + 1)))

    print('----------------------------')
    problem_name = abspath(ramp_kit_dir).split('/')[-1]
    print('Testing if the notebook can be converted to html')
    system('jupyter nbconvert --to html {}/{}_starting_kit.ipynb'.format(
        ramp_kit_dir, problem_name))
