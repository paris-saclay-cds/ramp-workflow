import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

# flake8: noqa
#import utils.classifier_with_metadata as wf

import rampwf as rw
from rampwf.utils import import_module_from_source

ss = StandardScaler()

class ClassifierWithMetaData(object):
    def __init__(self,target_names, workflow_element_names=['classifier']):
        self.element_names = workflow_element_names
        # self.name = 'classifier_workflow'  # temporary
        self.target_column_name = target_names

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        classifier = import_module_from_source(
            os.path.join(module_path, 'classifier.py'),
            self.element_names[0],
            sanitize=False
        )
        data_dtypes = X_array[1]
        data_dict = X_array[2]
        cols = X_array[3]
        X_array = X_array[0].copy()
        truth_names = ['y_' + t for t in self.target_column_name]
        X_array.drop(columns=truth_names, inplace=True)
        X_array = X_array.loc[train_is, :].values
        y_array = y_array[train_is]
        clf = classifier.Classifier(data_dtypes, data_dict, cols)
        clf.fit(X_array, y_array.ravel())
        return clf



    def test_submission(self, trained_model, X_array):
        clf = trained_model

        X_array = X_array[0].copy()
        truth_names = ['y_' + t for t in self.target_column_name]
        y = X_array[truth_names].values
        X_array.drop(columns=truth_names, inplace=True)
        X_array = X_array.values

        y_proba = clf.predict_proba(X_array, y)

        return y_proba

problem_title = 'Forest Cover Type Prediction'
_target_column_name = ['target']
_prediction_label_names = [0,1]

col_transformer = ColumnTransformer(transformers=[])

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = ClassifierWithMetaData(_target_column_name)

score_types = [
    rw.score_types.Accuracy(name='mean_score'),
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
]


def get_cv(X, y, train_sizes=None):
    cvs = []
    if train_sizes is None:
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=57)
        return list(cv.split(X[0], y))
    for train_size in train_sizes:
        cv = StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=train_size, random_state=57)
        cvs.append(next(cv.split(X[0], y)))
    print("cvs", cvs)
    return cvs


def _read_data(path, f_name, data_label, is_train):

    if data_label is None:
        data = pd.read_csv(os.path.join(path, 'data', f_name))
        dtypes = open(os.path.join(path, 'data', 'dtypes.json'))
    else:
        data = pd.read_csv(os.path.join(path, 'data', data_label, f_name))
        dtypes = open(os.path.join(path, 'data', data_label, 'dtypes.json'))

    dtypes_dict = json.loads(dtypes.read())
    dtypes.close()
    if 'target' not in data.columns[:2] :
        data = data[list(data.columns[:2]) + ['target']]
    else:
        data = data[list(data.columns[:2])]
    nums = [k for k, v in dtypes_dict.items() if v == "num" and k in data.columns]
    cats = [k for k, v in dtypes_dict.items() if v == "cat" and k in data.columns] + ['target']
    if is_train:
        col_transformer.set_params(transformers=[
            ('scaling', StandardScaler(), nums)
            , ('passthough', FunctionTransformer(lambda x: x, lambda x: x), cats)])
        scaled_data = col_transformer.fit_transform(data)


    else:
        scaled_data = col_transformer.transform(data)
    data = pd.DataFrame(scaled_data, index=data.index, columns=nums + cats)

    y_array = data[_target_column_name].values
    X_array = data.rename(columns={t: 'y_' + t for t in _target_column_name})
    dtypes = [dtypes_dict[col] for col in X_array.columns
              if col != 'y_target']
    cols = [col for col in X_array.columns if col != 'y_target']
    return (X_array, dtypes, dtypes_dict, cols), y_array


def get_train_data(path='.', data_label=None):
    f_name = 'train.csv'
    return _read_data(path, f_name, data_label, is_train=True)


def get_test_data(path='.', data_label = None):
    f_name = 'test.csv'
    return _read_data(path, f_name, data_label, is_train=False)
