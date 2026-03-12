import os
import copy

import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title = 'Boston housing price generative regression'
_target_column_names = ['lstat', 'medv']

MAX_COMPONENTS = 10

Predictions = rw.prediction_types.make_generative_regression(
    MAX_COMPONENTS, label_names=_target_column_names)

score_types = [
    rw.score_types.MDLikelihoodRatio(),
    rw.score_types.MDOutlierRate(),
    rw.score_types.MDR2(),
    rw.score_types.MDKSCalibration(),
]
# generate scores for each output dimension
_score_types = copy.deepcopy(score_types)
for o_i, o in enumerate(_target_column_names):
    dim_score_types = copy.deepcopy(_score_types)
    for score_type in dim_score_types:
        score_type.name = '{}_{}'.format(o, score_type.name)
        score_type.output_dim = o_i
    score_types += dim_score_types

workflow = rw.workflows.GenerativeRegressor(
    _target_column_names, MAX_COMPONENTS, check_sizes=[132], check_indexs=[13])


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=57)
    return cv.split(X)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_names].values
    X_array = data.rename(columns={t: "y_" + t for t in _target_column_names})
    return X_array, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
