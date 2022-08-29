import os
import pytest
from rampwf.hyperopt import Hyperparameter, run_hyperopt

PATH = os.path.dirname(__file__)

# flake8: noqa: E501


def test_hyperopt():
    submission = "rf"
    data_label = "cover_type_500"
    engine = "ray_hebo"

    ramp_kit_dir = os.path.join(PATH, "test_kit")
    run_hyperopt(
        ramp_kit_dir,
        ramp_kit_dir,
        os.path.join(ramp_kit_dir, "submissions"),
        data_label,
        submission,
        engine,
        1,
        True,
        True,
        True,
        True
    )