import os
import pandas as pd
import numpy as np
from rampwf.utils.importing import import_file
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types.base import BaseScoreType


# End of new score type

problem_title = 'Boston housing price regression'
_target_column_name = 'medv'
# A type (class) which will be used to create wrapper objects for y_pred

NB_BINS = 10

Predictions = rw.prediction_types.make_generative_regression(NB_BINS, label_names=['1'])

score_types = [
    rw.score_types.logLKGenerative(NB_BINS),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=57)
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


class GenerativeRegressor(object):
    def __init__(self, workflow_element_names=['gen_regressor']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        gen_regressor = import_file(module_path, self.element_names[0])
        reg = gen_regressor.GenerativeRegressor(NB_BINS)


        shape = y_array.shape
        if len(shape) == 1:
            y_array = y_array.reshape(-1, 1)
        elif len(shape) == 2:
            pass
        else:
            raise ValueError("More than two dims for y not supported")

        reg.fit(X_array[train_is], y_array[train_is])
        return reg

    def test_submission(self, trained_model, X_array):
        reg = trained_model
        y_pred, bin_edges = reg.predict(X_array)

        # All the bins are the same in this classification setup
        return np.concatenate((bin_edges, y_pred), axis=2)


# An object implementing the workflow
workflow = GenerativeRegressor()
