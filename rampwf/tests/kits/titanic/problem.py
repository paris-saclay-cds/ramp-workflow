import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import rampwf as rw

problem_title = 'Titanic survival classification'
_target_column_name = 'Survived'
_ignore_column_names = ['PassengerId']
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.SKLearnPipeline(file_name='submission', is_proba=True)

score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
    rw.score_types.ClassificationError(name='error'),
    rw.score_types.F1Above(name='f1_70', threshold=0.7),
    rw.score_types.BrierSkillScore(name="BSS", precision=3),
    rw.score_types.BrierScore(name="BS", precision=3),
    rw.score_types.BrierScoreReliability(name="BS Rel", precision=5),
    rw.score_types.BrierScoreResolution(name="BS Res", precision=3),
    rw.score_types.NormalizedGini(name='ngini', precision=3),
    rw.score_types.BalancedAccuracy(name='bac', precision=3),
    rw.score_types.MacroAveragedRecall(name='mar', precision=3),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
