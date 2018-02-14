import os

from rampwf.utils.testing import assert_submission, assert_notebook


PATH = os.path.dirname(__file__)


def test_notebook_testing():
    assert_notebook(ramp_kit_dir=os.path.join(PATH, "kits", "iris"))


def test_iris():

    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        submission='starting_kit', is_pickle=True,
        save_y_preds=True, retrain=True)


def test_boston_housing():

    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "boston_housing"),
        ramp_data_dir=os.path.join(PATH, "kits", "boston_housing"),
        submission='starting_kit', is_pickle=True,
        save_y_preds=False, retrain=False)


def test_el_nino():

    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "el_nino"),
        ramp_data_dir=os.path.join(PATH, "kits", "el_nino"),
        submission='starting_kit', is_pickle=True,
        save_y_preds=False, retrain=False)


def test_air_passengers():

    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "air_passengers"),
        ramp_data_dir=os.path.join(PATH, "kits", "air_passengers"),
        submission='starting_kit', is_pickle=True,
        save_y_preds=False, retrain=False)
