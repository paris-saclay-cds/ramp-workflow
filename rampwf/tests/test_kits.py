import os

from rampwf.utils.testing import assert_submission


PATH = os.path.dirname(__file__)


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
        save_y_preds=True, retrain=True)
