import os
import glob

import pytest

from rampwf.utils.testing import (
    assert_submission, assert_notebook, blend_submissions)


PATH = os.path.dirname(__file__)


def _generate_grid_path_kits():
    return [(os.path.abspath(path_kit,))
            for path_kit in sorted(glob.glob(os.path.join(PATH, 'kits', '*')))]


@pytest.mark.parametrize(
    "path_kit",
    _generate_grid_path_kits()
)
def test_notebook_testing(path_kit):
    # check if there is a notebook to be tested
    if len(glob.glob(os.path.join(path_kit, '*.ipynb'))):
        print(path_kit)
        assert_notebook(ramp_kit_dir=path_kit)


@pytest.mark.parametrize(
    "path_kit",
    _generate_grid_path_kits()
)
def test_submission(path_kit):
    submissions = sorted(glob.glob(os.path.join(path_kit, 'submissions', '*')))
    for sub in submissions:
        assert_submission(
            ramp_kit_dir=path_kit,
            ramp_data_dir=path_kit,
            submission=os.path.basename(sub), is_pickle=True,
            save_y_preds=False, retrain=False)


def test_blending():
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        submission='starting_kit', is_pickle=True,
        save_y_preds=True, retrain=True)
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        submission='random_forest_10_10', is_pickle=True,
        save_y_preds=True, retrain=True)
    blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"))
