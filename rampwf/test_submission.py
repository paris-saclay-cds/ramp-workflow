# -*- coding: utf-8 -*-
from __future__ import print_function
import imp
import numpy as np

problem = imp.load_source('', 'problem.py')
print('Testing {}'.format(problem.problem_title))
print('Reading file ...')
X, y = problem.get_data()
prediction_labels = problem.prediction_labels
score_types = problem.score_types
print('Training model ...')
cv = list(problem.get_cv(X, y))
module_path = 'submissions/starting_kit'
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
