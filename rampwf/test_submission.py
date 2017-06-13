# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import imp
import numpy as np

# Command line parameters:
#     path=<path>  (default: .)
#     name=<submission_name> (default: starting_kit)
# It will look for problem.py in <path> and workflow elements (submission)
# in <path>/submissions/<submission_name>
path = '.'
submission_name = 'starting_kit'
for arg in sys.argv[1:]:
    tokens = arg.split('=')
    if tokens[0] == 'path':
        path = tokens[1]
    elif tokens[0] == 'submission':
        submission_name = tokens[1]
    else:
        print('Unknown argument {}'.format(tokens[0]))
        exit(0)


problem = imp.load_source('', os.path.join(path, 'problem.py'))
print('Testing {} backend'.format(problem.problem_title))
print('Reading file ...')
X_train, y_train = problem.get_train_data(path=path)
X_test, y_test = problem.get_test_data(path=path)
prediction_labels = problem.prediction_labels
score_types = problem.score_types
print('Training {} in {}/submissions ...'.format(submission_name, path))
cv = list(problem.get_cv(X_train, y_train))
module_path = os.path.join(path, 'submissions', submission_name)
train_train_scoress = np.empty((len(cv), len(score_types)))
train_valid_scoress = np.empty((len(cv), len(score_types)))
test_scoress = np.empty((len(cv), len(score_types)))
for fold_i, (train_is, valid_is) in enumerate(cv):
    trained_workflow = problem.workflow.train_submission(
        module_path, X_train, y_train, train_is=train_is)

    y_pred_train = problem.workflow.test_submission(trained_workflow, X_train)
    predictions_train_train = problem.prediction_type.Predictions(
        y_pred=y_pred_train[train_is], labels=prediction_labels)
    ground_truth_train_train = problem.prediction_type.Predictions(
        y_true=y_train[train_is], labels=prediction_labels)
    predictions_train_valid = problem.prediction_type.Predictions(
        y_pred=y_pred_train[valid_is], labels=prediction_labels)
    ground_truth_train_valid = problem.prediction_type.Predictions(
        y_true=y_train[valid_is], labels=prediction_labels)

    y_pred_test = problem.workflow.test_submission(trained_workflow, X_test)
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
