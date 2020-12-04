import pytest
import numpy as np

from rampwf.score_types.generative_regression import (
    MDNegativeLogLikelihood, MDLikelihoodRatio, MDRMSE,
    MDR2, MDKSCalibration, MDOutlierRate)
from rampwf.utils import MixtureYPred

WEIGHTS = np.array([[1.0], ] * 2)

TYPES = np.array([[0.0], ] * 2)
PARAMS_1 = np.array([[0.55, 0.1], ] * 2)
PARAMS_2 = np.array([[0.0, 0.1], ] * 2)
Y_PRED_1 = MixtureYPred().add(WEIGHTS, TYPES, PARAMS_1).finalize()
Y_PRED_2 = MixtureYPred().add(WEIGHTS, TYPES, PARAMS_2).finalize()

TYPES_UNI = np.array([[1.0], ] * 2)
PARAMS_UNI_1 = np.array([[0.49, 0.03], ] * 2)
Y_PRED_UNI_1 = MixtureYPred().add(WEIGHTS, TYPES_UNI, PARAMS_UNI_1).finalize()
PARAMS_UNI_2 = np.array([[0.45, 0.05], ] * 2)
Y_PRED_UNI_2 = MixtureYPred().add(WEIGHTS, TYPES_UNI, PARAMS_UNI_2).finalize()
Y_TRUTH_1 = np.array([[0.5], [0.51]])


def test_likelihood():
    """
    Basic tests to check the negative log likelihood and likelihood ratio
    with normal and uniform distribution.
    """
    n_log = MDNegativeLogLikelihood()
    assert n_log(Y_TRUTH_1, Y_PRED_1) < n_log(Y_TRUTH_1, Y_PRED_2)
    assert n_log(Y_TRUTH_1, Y_PRED_2) == pytest.approx(11.3688534, 10e-5)
    assert n_log(Y_TRUTH_1, Y_PRED_UNI_1) == pytest.approx(-3.506557, 10e-5)

    n_log_ratio = MDLikelihoodRatio()
    assert n_log_ratio(Y_TRUTH_1, Y_PRED_1) > n_log_ratio(Y_TRUTH_1, Y_PRED_2)


def test_metrics():
    """
    Basic tests to make sure the metrics on generative regression are working.
    """
    rmse = MDRMSE()
    assert rmse(Y_TRUTH_1, Y_PRED_UNI_1) == pytest.approx(0.005, 10e-5)

    mdr = MDR2()
    assert mdr(Y_TRUTH_1, Y_PRED_1) == pytest.approx(-81, 10e-5)

    mdks = MDKSCalibration()
    assert mdks(Y_TRUTH_1, Y_PRED_UNI_1) == pytest.approx(1/3, 10e-5)

    mdout = MDOutlierRate()
    assert mdout(Y_TRUTH_1, Y_PRED_UNI_2) == pytest.approx(0.5, 10e-5)
