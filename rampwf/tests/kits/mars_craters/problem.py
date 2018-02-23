import os

import numpy as np
import pandas as pd

import rampwf as rw


problem_title = 'Mars craters detection and classification'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_detection()
# An object implementing the workflow
workflow = rw.workflows.ObjectDetector()

# The overlap between adjacent patches is 56 pixels
# The scoring region is chosen so that despite the overlap,
# no crater is scored twice, hence the boundaries of
# 28 = 56 / 2 and 196 = 224 - 56 / 2
minipatch = [28, 196, 28, 196]

score_types = [
    rw.score_types.OSPA(minipatch=minipatch),
    rw.score_types.SCP(shape=(224, 224), minipatch=minipatch),
    rw.score_types.DetectionAveragePrecision(name='ap'),
    rw.score_types.DetectionPrecision(
        name='prec(0)', iou_threshold=0.0, minipatch=minipatch),
    rw.score_types.DetectionPrecision(
        name='prec(0.5)', iou_threshold=0.5, minipatch=minipatch),
    rw.score_types.DetectionPrecision(
        name='prec(0.9)', iou_threshold=0.9, minipatch=minipatch),
    rw.score_types.DetectionRecall(
        name='rec(0)', iou_threshold=0.0, minipatch=minipatch),
    rw.score_types.DetectionRecall(
        name='rec(0.5)', iou_threshold=0.5, minipatch=minipatch),
    rw.score_types.DetectionRecall(
        name='rec(0.9)', iou_threshold=0.9, minipatch=minipatch),
    rw.score_types.MADCenter(name='madc', minipatch=minipatch),
    rw.score_types.MADRadius(name='madr', minipatch=minipatch)
]


def get_cv(X, y):
    # 3 quadrangles for training have not exactly the same size,
    # but for simplicity just cut in 3
    # for each fold use one quadrangle as test set, the other two as training

    n_tot = len(X)
    n1 = n_tot // 3
    n2 = n1 * 2

    return [(np.r_[0:n2], np.r_[n2:n_tot]),
            (np.r_[n1:n_tot], np.r_[0:n1]),
            (np.r_[0:n1, n2:n_tot], np.r_[n1:n2])]


def _read_data(path, typ):
    """
    Read and process data and labels.

    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'train', 'test'}

    Returns
    -------
    X, y data

    """

    suffix = '_mini'

    try:
        data_path = os.path.join(path, 'data',
                                 'data_{0}{1}.npy'.format(typ, suffix))
        src = np.load(data_path, mmap_mode='r')

        labels_path = os.path.join(path, 'data',
                                   'labels_{0}{1}.csv'.format(typ, suffix))
        labels = pd.read_csv(labels_path)
    except IOError:
        raise IOError("'data/data_{0}.npy' and 'data/labels_{0}.csv' are not "
                      "found. Ensure you ran 'python download_data.py' to "
                      "obtain the train/test data".format(typ))

    # convert the dataframe with crater positions to list of
    # list of (x, y, radius) tuples (list of arrays of shape (n, 3) with n
    # true craters on an image

    # determine locations of craters for each patch in the labels array
    n_true_patches = labels.groupby('i').size().reindex(
        range(src.shape[0]), fill_value=0).values
    # make cumulative sum to obtain start/stop to slice the labels
    n_cum = np.array(n_true_patches).cumsum()
    n_cum = np.insert(n_cum, 0, 0)

    labels_array = labels[['row_p', 'col_p', 'radius_p']].values
    y = [[tuple(x) for x in labels_array[i:j]]
         for i, j in zip(n_cum[:-1], n_cum[1:])]
    # convert list to object array of lists
    y_array = np.empty(len(y), dtype=object)
    y_array[:] = y

    return src, y_array


def get_test_data(path='.'):
    return _read_data(path, 'test')


def get_train_data(path='.'):
    return _read_data(path, 'train')
