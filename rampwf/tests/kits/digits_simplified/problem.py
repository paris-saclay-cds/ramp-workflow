import os

import numpy as np
import pandas as pd
import rampwf as rw

from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

from skimage.io import imsave

problem_title = 'Digits classification'
_target_column_name = 'class'
_prediction_label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.SimplifiedImageClassifier(
    n_classes=len(_prediction_label_names),
)

score_types = [
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
]


def get_cv(folder_X, y):
    _, X = folder_X
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df['id'].values
    y = df['class'].values
    folder = os.path.join(path, 'data', 'imgs')
    return (folder, X), y


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


###############################################################################
# This section create the images to be used for classification.
# Note that it should be included in normal kit.

KIT_DIR = os.path.dirname(__file__)

digits = load_digits()

data_dir = os.path.join(KIT_DIR, 'data')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
n_images = digits.data.shape[0]
n_train_images = int(n_images * 0.8)

img_dir = os.path.join(data_dir, 'imgs')
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

filenames_image = []
for img_idx, img in zip(range(n_images), digits.data):
    filename = os.path.join(img_dir, str(img_idx) + '.png')
    imsave(filename, img.reshape((8, 8)).astype(np.int8))
    filenames_image.append(filename)

train_csv = pd.DataFrame({'id': np.array(filenames_image[:n_train_images]),
                          'class': digits.target[:n_train_images]})
train_csv = train_csv.set_index('id')
train_csv.to_csv(os.path.join(data_dir, 'train.csv'))

test_csv = pd.DataFrame({'id': np.array(filenames_image[n_train_images:]),
                         'class': digits.target[n_train_images:]})
test_csv = test_csv.set_index('id')
test_csv.to_csv(os.path.join(data_dir, 'test.csv'))
