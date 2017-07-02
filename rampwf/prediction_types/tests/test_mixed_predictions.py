"""Testing for mixed predictions (rampwf.prediction.mixed)."""

# Author: Balazs Kegl <balazs.kegl@gmail.com>
# License: BSD 3 clause

# import numpy as np
# from rampwf.prediction_types.mixed import Predictions
# from numpy.testing import assert_array_equal


# def test_init():
#     labels = ['Class_1', 'Class_2', 'Class_3']
#     ps_1 = [0.7, 0.1, 0.2, 1.85]
#     ps_2 = [0.1, 0.1, 0.8, 0.15]
#     ps_3 = [0.2, 0.5, 0.3, -0.22]
#     predictions = Predictions(
#         labels=labels, y_pred=np.array([ps_1, ps_2, ps_3]))
#     assert_array_equal(predictions.y_pred, [ps_1, ps_2, ps_3])
#     assert_array_equal(predictions.multiclass.y_pred_label_index, [0, 2, 1])
#     assert_array_equal(
#         predictions.multiclass.y_pred, [ps_1[:-1], ps_2[:-1], ps_3[:-1]])
#     assert_array_equal(
#         predictions.regression.y_pred, [ps_1[-1], ps_2[-1], ps_3[-1]])


# def test_init_empty():
#     labels = ['Class_1', 'Class_2', 'Class_3']
#     predictions = Predictions(labels=labels, shape=(4, 4))
#     assert_array_equal(predictions.y_pred, np.array([
#         [np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan],
#     ]))


# def test_set_valid_in_train():
#     labels = ['Class_1', 'Class_2', 'Class_3']
#     predictions = Predictions(labels=labels, shape=(4, 4))
#     assert_array_equal(predictions.y_pred, np.array([
#         [np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan],
#     ]))
#     ps_1 = [0.7, 0.1, 0.2, 1.85]
#     ps_2 = [0.1, 0.1, 0.8, 0.15]
#     ps_3 = [0.2, 0.5, 0.3, -0.22]
#     predictions_valid = Predictions(
#         labels=labels, y_pred=np.array([ps_1, ps_2, ps_3]))
#     predictions.set_valid_in_train(predictions_valid, [0, 2, 3])
#     assert_array_equal(predictions.y_pred, np.array([
#         [0.7, 0.1, 0.2, 1.85],
#         [np.nan, np.nan, np.nan, np.nan],
#         [0.1, 0.1, 0.8, 0.15],
#         [0.2, 0.5, 0.3, -0.22],
#     ]))
