from rampwf.utils.testing import assert_submission

import os



_module_path = os.path.dirname(__file__)


def test_iris():

    assert_submission(
                    ramp_kit_dir=os.path.join(_module_path, "kits", "iris"),
                    ramp_data_dir=os.path.join(_module_path, "kits", "iris"),
                    submission='starting_kit', is_pickle=True,
                    save_y_preds=True, retrain=True)