from __future__ import division

import numpy as np

from rampwf.score_types.soft_accuracy import SoftAccuracy


score_matrix_1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

score_matrix_2 = np.array([
    [1, 0.5, 0],
    [0.3, 1, 0.3],
    [0, 0.5, 1],
])

y_true_proba_1 = np.array([1, 0, 0])
y_true_proba_2 = np.array([0, 1, 0])
y_true_proba_3 = np.array([0.5, 0.5, 0])

y_proba_1 = np.array([1, 0, 0])
y_proba_2 = np.array([0, 1, 0])


def test_soft_accuracy():
    score_1 = SoftAccuracy(score_matrix=score_matrix_1)
    assert score_1(np.array([y_true_proba_1]), np.array([y_proba_1])) == 1
    assert score_1(np.array([y_true_proba_1]), np.array([y_proba_2])) == 0
    assert score_1(np.array([y_true_proba_2]), np.array([y_proba_1])) == 0
    assert score_1(np.array([y_true_proba_2]), np.array([y_proba_2])) == 1
    assert score_1(np.array([y_true_proba_3]), np.array([y_proba_1])) == 0.5
    assert score_1(np.array([y_true_proba_3]), np.array([y_proba_2])) == 0.5

    score_2 = SoftAccuracy(score_matrix=score_matrix_2)
    assert score_2(np.array([y_true_proba_1]), np.array([y_proba_1])) == 1
    assert score_2(np.array([y_true_proba_1]), np.array([y_proba_2])) == 0.5
    assert score_2(np.array([y_true_proba_2]), np.array([y_proba_1])) == 0.3
    assert score_2(np.array([y_true_proba_2]), np.array([y_proba_2])) == 1
    assert score_2(np.array([y_true_proba_3]), np.array([y_proba_1])) == 0.65
    assert score_2(np.array([y_true_proba_3]), np.array([y_proba_2])) == 0.75
