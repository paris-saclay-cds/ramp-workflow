"""RAMP definition for the drug spectra problem.

This script demonstrates how to define a multi-objective RAMP where
scores are defined and displayed on various slices of the output y_pred
and a combined score is defined as a weighted combination of the individual
scores.

"""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import rampwf as rw


problem_title =\
    'Drug classification and concentration estimation from Raman spectra'
# label names for the classification target
_prediction_label_names = ['A', 'B', 'Q', 'R']
# the regression target column
_target_column_name_clf = 'molecule'
# the classification target column
_target_column_name_reg = 'concentration'

# The first four columns of y_pred will be wrapped in multiclass Predictions.
Predictions_1 = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# The last column of y_pred will be wrapped in regression Predictions.
# We make a 2D but single-column y_pred (instead of a classical 1D y_pred)
# to make handling the combined 2D y_pred array easier.
Predictions_2 = rw.prediction_types.make_regression(
    label_names=[_target_column_name_reg])
# The combined Predictions is initialized by the list of individual Predictions.
Predictions = rw.prediction_types.make_combined([Predictions_1, Predictions_2])

# The workflow object, named after the RAMP.
workflow = rw.workflows.DrugSpectra()

# The first score will be applied on the first Predictions
score_type_1 = rw.score_types.ClassificationError(name='err', precision=3)
# The second score will be applied on the second Predictions
score_type_2 = rw.score_types.MARE(name='mare', precision=3)
score_types = [
    # The official score combines the two scores with weights 2/3 and 1/3.
    rw.score_types.Combined(
        name='combined', score_types=[score_type_1, score_type_2],
        weights=[2. / 3, 1. / 3], precision=3),
    # To let the score type know that it should be applied on the first
    # Predictions of the combined Predictions' prediction_list, we wrap
    # it into a special MakeCombined score with index 0
    rw.score_types.MakeCombined(score_type=score_type_1, index=0),
    rw.score_types.MakeCombined(score_type=score_type_2, index=1),
]


def _read_data(path, f_name):
    X_df = pd.read_csv(os.path.join(path, 'data', f_name))
    y_columns = [_target_column_name_clf, _target_column_name_reg]
    y_array = X_df[y_columns].values
    X_df = X_df.drop(y_columns, axis=1)
    # Convert spectra entry from string to array of floats
    X_df['spectra'] = X_df.spectra.apply(
        lambda x: np.fromstring(x[1:-1], sep=',', dtype=float))

    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train_mini.csv.bz2'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test_mini.csv.bz2'
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=57)
    return cv.split(X)
