from __future__ import division

import numpy as np

from rampwf.score_types.generative_regression import (
    MDNegativeLogLikelihood, MDLikelihoodRatio, MDRMSE,
    MDR2, MDKSCalibration, MDOutlierRate)
from rampwf.utils import MixtureYPred
import pytest

weights = np.array([[1.0], ] * 2)

types = np.array([[0.0], ] * 2)
params_1 = np.array([[0.55, 0.1], ] * 2)
params_2 = np.array([[0.0, 0.1], ] * 2)
y_pred_1 = MixtureYPred().add(weights, types, params_1).finalize()
y_pred_2 = MixtureYPred().add(weights, types, params_2).finalize()

types_uni = np.array([[1.0], ] * 2)
params_uni_1 = np.array([[0.49, 0.52], ] * 2)
y_pred_uni_1 = MixtureYPred().add(weights, types_uni, params_uni_1).finalize()
params_uni_2 = np.array([[0.45, 0.5], ] * 2)
y_pred_uni_2 = MixtureYPred().add(weights, types_uni, params_uni_2).finalize()
y_truth_1 = np.array([[0.5], [0.51]])


def test_likelihood():
    n_log = MDNegativeLogLikelihood()
    assert n_log(y_truth_1, y_pred_1) < n_log(y_truth_1, y_pred_2)
    assert n_log(y_truth_1, y_pred_2) == pytest.approx(11.3688534, 10e-5)
    assert n_log(y_truth_1, y_pred_uni_1) == pytest.approx(-3.506557, 10e-5)

    n_log_ratio = MDLikelihoodRatio()
    assert n_log_ratio(y_truth_1, y_pred_1) > n_log_ratio(y_truth_1, y_pred_2)


def test_metrics():
    rmse = MDRMSE()
    assert rmse(y_truth_1, y_pred_uni_1) == pytest.approx(0.005, 10e-5)

    mdr = MDR2()
    assert mdr(y_truth_1, y_pred_1) == pytest.approx(-81, 10e-5)

    mdks = MDKSCalibration()
    assert mdks(y_truth_1, y_pred_uni_1) == pytest.approx(1/3, 10e-5)

    mdout = MDOutlierRate()
    assert mdout(y_truth_1, y_pred_uni_2) == pytest.approx(0.5, 10e-5)
