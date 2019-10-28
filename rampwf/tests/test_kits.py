import os
import glob

import pytest

from rampwf.utils.testing import (
    assert_submission, assert_notebook, blend_submissions)


PATH = os.path.dirname(__file__)


def skip_no_tensorflow():
    try:
        import tensorflow  # noqa
    except ImportError:
        return pytest.mark.skip(reason='tensorflow not available')
    return pytest.mark.basic


def _generate_grid_path_kits():
    grid = []
    for path_kit in sorted(glob.glob(os.path.join(PATH, 'kits', '*'))):
        if 'digits' in path_kit:
            grid.append(pytest.param(os.path.abspath(path_kit),
                                     marks=skip_no_tensorflow()))
        else:
            grid.append(os.path.abspath(path_kit))
    return grid


@pytest.mark.parametrize(
    "path_kit",
    _generate_grid_path_kits()
)
def test_notebook_testing(path_kit):
    # check if there is a notebook to be tested
    print('pytttttt')
    print(path_kit)
    print(os.path.join(path_kit, '*.ipynb'))
    print(len(glob.glob(os.path.join(path_kit, '*.ipynb'))))
    
    if len(glob.glob(os.path.join(path_kit, '*.ipynb'))):
        print('PATH IS: ',path_kit)
        assert_notebook(ramp_kit_dir=path_kit)


@pytest.mark.parametrize(
    "path_kit",
    _generate_grid_path_kits()
)
def test_submission(path_kit):
    submissions = sorted(glob.glob(os.path.join(path_kit, 'submissions', '*')))
    for sub in submissions:
        # FIXME: to be removed once el-nino tests is fixed.
        if 'el_nino' in sub:
            pytest.xfail('el-nino is failing due to xarray.')
        else:
            assert_submission(
                ramp_kit_dir=path_kit,
                ramp_data_dir=path_kit,
                ramp_submission_dir=os.path.join(path_kit, 'submissions'),
                submission=os.path.basename(sub), is_pickle=True,
                save_output=False, retrain=True)


def test_blending():
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        submission='starting_kit', is_pickle=True,
        save_output=True, retrain=True)
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        submission='random_forest_10_10', is_pickle=True,
        save_output=True, retrain=True)
    blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        save_output=True)
