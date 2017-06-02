"""Testing for multiclass predictions (rampwf.prediction.multiclass)."""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

import numpy as np
from rampwf.prediction_types.multiclass import Predictions
from numpy.testing import assert_array_equal


def test_init():
    labels = ['Class_1', 'Class_2', 'Class_3']
    ps_1 = [0.7, 0.1, 0.2]
    ps_2 = [0.1, 0.8, 0.1]
    ps_3 = [0.2, 0.5, 0.3]
    predictions = Predictions(labels=labels, y_pred=[ps_1, ps_2, ps_3])
    assert_array_equal(predictions.y_pred, [ps_1, ps_2, ps_3])
    assert_array_equal(predictions.y_pred_label_index, [0, 1, 1])


def test_init_empty():
    labels = ['Class_1', 'Class_2', 'Class_3']
    predictions = Predictions(labels=labels, shape=(4,))
    assert_array_equal(predictions.y_pred, np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ]))
    predictions = Predictions(labels=labels, shape=(4, 3))
    assert_array_equal(predictions.y_pred, np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ]))


def test_init_from_labels():
    labels = ['Class_1', 'Class_2', 'Class_3']
    y_pred_label_1 = 'Class_2'
    y_pred_label_2 = 'Class_1'
    y_pred_label_3 = 'Class_3'
    predictions = Predictions(
        labels=labels,
        y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


def test_init_from_truth():
    labels = ['Class_1', 'Class_2', 'Class_3']
    y_pred_label_1 = 'Class_2'
    y_pred_label_2 = 'Class_1'
    y_pred_label_3 = 'Class_3'
    predictions = Predictions(
        labels=labels,
        y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


def test_list_init_from_labels():
    labels = ['Class_1', 'Class_2', 'Class_3']
    y_pred_label_1 = ['Class_2']
    y_pred_label_2 = ['Class_1']
    y_pred_label_3 = ['Class_3']
    predictions = Predictions(
        labels=labels,
        y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


def test_list_init_from_truth():
    labels = ['Class_1', 'Class_2', 'Class_3']
    y_pred_label_1 = ['Class_2']
    y_pred_label_2 = ['Class_1']
    y_pred_label_3 = ['Class_3']
    predictions = Predictions(
        labels=labels,
        y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


def test_multiclass_init():
    labels = [1, 2, 3]
    y_pred_label_1 = [2]
    y_pred_label_2 = [1]
    y_pred_label_3 = [3]
    predictions = Predictions(
        labels=labels,
        y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


def test_multiclass_init_from_truth():
    labels = [1, 2, 3]
    y_pred_label_1 = [2]
    y_pred_label_2 = [1]
    y_pred_label_3 = [3]
    predictions = Predictions(
        labels=labels,
        y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


def test_multiclass_init_from_labels():
    labels = [1, 2, 3]
    y_pred_label_1 = 2
    y_pred_label_2 = 1
    y_pred_label_3 = 3
    predictions = Predictions(
        labels=labels,
        y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


def test_multilabel_init():
    labels = [1, 2, 3]
    y_pred_label_1 = [1, 3]
    y_pred_label_2 = [2]
    y_pred_label_3 = [1, 2, 3]
    predictions = Predictions(
        labels=labels,
        y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred,
                       [[0.5, 0, 0.5], [0, 1, 0], [1.0 / 3, 1.0 / 3, 1.0 / 3]])
    # in case of ties, argmax returns the index of the first max
    assert_array_equal(predictions.y_pred_label_index, [0, 1, 0])


def test_multilabel_init_from_truth():
    labels = [1, 2, 3]
    y_pred_label_1 = [1, 3]
    y_pred_label_2 = [2]
    y_pred_label_3 = [1, 2, 3]
    predictions = Predictions(
        labels=labels,
        y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    assert_array_equal(predictions.y_pred,
                       [[0.5, 0, 0.5], [0, 1, 0], [1.0 / 3, 1.0 / 3, 1.0 / 3]])
    # in case of ties, argmax returns the index of the first max
    assert_array_equal(predictions.y_pred_label_index, [0, 1, 0])


def test_set_valid_in_train():
    labels = ['Class_1', 'Class_2', 'Class_3']
    predictions = Predictions(labels=labels, shape=(4,))
    y_pred_label_1 = ['Class_2']
    y_pred_label_2 = ['Class_1']
    y_pred_label_3 = ['Class_3']
    predictions_valid = Predictions(
        labels=labels,
        y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
    predictions.set_valid_in_train(predictions_valid, [0, 2, 3])
    assert_array_equal(predictions.y_pred, np.array([
        [0, 1, 0],
        [np.nan, np.nan, np.nan],
        [1, 0, 0],
        [0, 0, 1],
    ]))
