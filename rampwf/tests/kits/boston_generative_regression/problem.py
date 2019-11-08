import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title = 'Boston housing price regression'
_target_column_name = ['medv']

MAX_DISTS = 10

Predictions = rw.prediction_types. \
    make_generative_regression(MAX_DISTS,
                               label_names=_target_column_name)

score_types = [
    rw.score_types.NegativeLogLikelihoodRegDists(),
    rw.score_types.LikelihoodRatioDists()
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=57)
    return cv.split(X)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_array = data.rename(columns={t: "y_" + t for t in _target_column_name})
    return X_array, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


workflow = rw.workflows.GenerativeRegressor(_target_column_name, MAX_DISTS,
                                            check_sizes=[132], check_indexs=[13])
