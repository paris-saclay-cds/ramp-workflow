import os

import pandas as pd
from sklearn.model_selection import ShuffleSplit

import rampwf as rw

problem_title = 'Boston housing price regression'
_target_column_name = 'medv'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
# workflow = rw.workflows.Regressor()
workflow = rw.workflows.Workflow(['regressor'])


score_types = [
    rw.score_types.RMSE(),
    rw.score_types.RelativeRMSE(name='rel_rmse'),
    rw.score_types.NormalizedRMSE(name='n_rmse'),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_array = data.drop([_target_column_name], axis=1).values
    return X_array, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
