import os
import pytest
from rampwf.hyperopt import Hyperparameter, run_hyperopt

PATH = os.path.dirname(__file__)

# flake8: noqa: E501


def test_hyperopt():
    # we can remove the creation of the folder
    # when only python >= 3.8 is supported by using
    # dirs_exist_ok
    submission = "rf"
    data = "cover_type_500"
    engine = "ray_hebo"

    ramp_kit_dir = os.path.join(PATH, "generative_classifier")
    destination_folder = os.path.join(
        ramp_kit_dir, "submissions", f"{submission}_{data}_{engine}_hyperopt"
    )
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    run_hyperopt(
        ramp_kit_dir,
        ramp_kit_dir,
        os.path.join(ramp_kit_dir, "submissions"),
        data,
        submission,
        engine,
        1,
        True,
        False,
        False,
        False
    )