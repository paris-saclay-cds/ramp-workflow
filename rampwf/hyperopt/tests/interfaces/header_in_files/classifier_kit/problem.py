import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Titanic survival classification'
_target_column_name = 'Survived'
_ignore_column_names = ['PassengerId']
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name, data_label):
    if data_label is not None:
        data = pd.read_csv(os.path.join(path, 'data', data_label, f_name))
    else:
        data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.', data_label=None):
    f_name = 'train.csv'
    return _read_data(path, f_name, data_label)


def get_test_data(path='.', data_label=None):
    f_name = 'test.csv'
    return _read_data(path, f_name, data_label)
