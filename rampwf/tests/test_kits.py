import os
import glob
import shutil
from textwrap import dedent

import cloudpickle

import pytest

import numpy as np

from rampwf.utils import import_module_from_source
from rampwf.utils.testing import assert_read_problem
from rampwf.utils.testing import assert_submission
from rampwf.utils.testing import assert_notebook
from rampwf.utils.testing import blend_submissions
from rampwf.utils.testing import bag_then_blend_submissions
from rampwf.utils.testing import assert_data


PATH = os.path.dirname(__file__)


def no_tensorflow():
    try:
        import tensorflow  # noqa
    except ImportError:
        return True
    return False


def _generate_grid_path_kits():
    grid = []
    for path_kit in sorted(glob.glob(os.path.join(PATH, 'kits', '*'))):
        if 'digits' in path_kit:
            marks = pytest.mark.skipif(no_tensorflow(),
                                       reason='tensorflow not available')
            grid.append(pytest.param(os.path.abspath(path_kit), marks=marks))
        elif 'data_label' in path_kit:
            pass
        else:
            grid.append(os.path.abspath(path_kit))
    return grid


def test_external_imports(tmpdir):
    # checking imports from an external_imports folder located in the
    # ramp_kit_dir

    # temporary kit
    path_kit = tmpdir.join("titanic_external_imports")
    shutil.copytree(os.path.join(PATH, "kits", "titanic"), path_kit)
    problem_path = os.path.join(path_kit, "problem.py")
    submissions_dir = os.path.join(path_kit, 'submissions')
    submission_path = os.path.join(submissions_dir, 'starting_kit')
    estimator_path = os.path.join(submission_path, "estimator.py")

    # module to be imported
    ext_module_dir = path_kit.mkdir("external_imports").mkdir("utils")
    with open(os.path.join(ext_module_dir, "test_imports.py"), 'w+') as f:
        f.write(
            dedent(
                """
                x = 2
                """
            )
        )

    for path in [problem_path, estimator_path]:
        with open(path, 'a') as f:
            f.write(
                dedent(
                    """
                    from utils import test_imports
                    assert test_imports.x == 2
                    """
                )
            )

    assert_submission(
        ramp_kit_dir=path_kit,
        ramp_data_dir=path_kit,
        ramp_submission_dir=submissions_dir,
        submission=submission_path,
        is_pickle=True,
        save_output=False,
        retrain=True)


@pytest.mark.skip(reason="skipping notebook tests for now.")
@pytest.mark.parametrize(
    "path_kit",
    _generate_grid_path_kits())
def test_notebook_testing(path_kit):
    # check if there is a notebook to be tested
    if len(glob.glob(os.path.join(path_kit, '*.ipynb'))):
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
                save_output=False, retrain=True
            )
        # testing non-consecutive fold_idxs
        # cv in kit needs at least 3 fold
        if  not 'acrobot' in sub:
            assert_submission(
                ramp_kit_dir=path_kit,
                ramp_data_dir=path_kit,
                ramp_submission_dir=os.path.join(path_kit, 'submissions'),
                submission=os.path.basename(sub), is_pickle=True,
                save_output=False, retrain=True,
                fold_idxs = [0, 2]
            )
        # testing the partial training workflow
        if 'titanic_old' in sub or 'air_passengers_old' in sub:
            assert_submission(
                ramp_kit_dir=path_kit,
                ramp_data_dir=path_kit,
                ramp_submission_dir=os.path.join(path_kit, 'submissions'),
                submission=os.path.basename(sub), is_pickle=True,
                is_partial_train=True,
                save_output=False, retrain=True
            )


def test_fe_gen_reg_numpy():
    # the numpy workflow should give the same output as the time series one using
    # pandas when n_burn_in = 0 and n_lookahead = 1
    acrobot_numpy_path = os.path.join(PATH, 'kits', 'acrobot_numpy')
    acrobot_ts_generative_regression_path = os.path.join(
        PATH, 'kits', 'acrobot_ts_generative_regression')

    kit_paths = [acrobot_numpy_path, acrobot_ts_generative_regression_path]
    submissions = [
        os.path.join(acrobot_numpy_path, 'submissions', 'starting_kit'),
        os.path.join(
            acrobot_ts_generative_regression_path,
            'submissions', 'starting_kit')
    ]

    y_preds = []
    for kit_path, submission in zip(kit_paths, submissions):
        problem = assert_read_problem(kit_path)
        X_train, y_train, X_test, _ = assert_data(
            kit_path, kit_path
        )
        trained_workflow = problem.workflow.train_submission(
            submission, X_train, y_train)
        y_pred = problem.workflow.test_submission(
            trained_workflow, X_test)
        y_preds.append(y_pred)

    np.testing.assert_array_almost_equal(y_preds[0], y_preds[1], decimal=8)


def test_blending():
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        submission='starting_kit', is_pickle=True,
        save_output=True, retrain=False,
        fold_idxs=[0, 2],
    )
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        submission='random_forest_10_10', is_pickle=True,
        save_output=True, retrain=False,
        fold_idxs=[0, 1]
    )
    blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        save_output=True
    )
    bag_then_blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        save_output=True
    )
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        submission='starting_kit', is_pickle=True,
        save_output=True, retrain=True
    )
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        submission='random_forest_10_10', is_pickle=True,
        save_output=True, retrain=True
    )
    blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        save_output=True
    )
    bag_then_blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        save_output=True
    )
    blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        save_output=True,
        fold_idxs=[0, 2]
    )
    bag_then_blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris"),
        ramp_submission_dir=os.path.join(PATH, "kits", "iris", "submissions"),
        save_output=True,
        fold_idxs=[0, 2]
    )
    # cleaning up so next test doesn't try to train "training_output"
    shutil.rmtree(os.path.join(
        PATH, "kits", "iris", "submissions", "training_output"))


def test_data_label():
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris_data_label"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris_data_label"),
        data_label='data_label',
        ramp_submission_dir=os.path.join(
            PATH, "kits", "iris_data_label", "submissions"),
        submission='starting_kit', is_pickle=True,
        save_output=True, retrain=True)
    assert_submission(
        ramp_kit_dir=os.path.join(PATH, "kits", "iris_data_label"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris_data_label"),
        data_label='data_label',
        ramp_submission_dir=os.path.join(
            PATH, "kits", "iris_data_label", "submissions"),
        submission='random_forest_10_10', is_pickle=True,
        save_output=True, retrain=True)
    blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris_data_label"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris_data_label"),
        data_label='data_label',
        ramp_submission_dir=os.path.join(
            PATH, "kits", "iris_data_label", "submissions"),
        save_output=True)
    bag_then_blend_submissions(
        ['starting_kit', 'random_forest_10_10'],
        ramp_kit_dir=os.path.join(PATH, "kits", "iris_data_label"),
        ramp_data_dir=os.path.join(PATH, "kits", "iris_data_label"),
        data_label='data_label',
        ramp_submission_dir=os.path.join(
            PATH, "kits", "iris_data_label", "submissions"),
        save_output=True)
    # cleaning up so next test doesn't try to train "training_output"
    shutil.rmtree(os.path.join(
        PATH, "kits", "iris_data_label", "submissions", "training_output"))


def test_cloudpickle():
    """Check cloudpickle works with the way modules are imported from source.

    This only checks that an object that can be pickled with cloudpickle can
    still be pickled with cloudpickle when imported dynamically using
    import_module_from_source.
    """
    # use iris_old as the object has to be a custom class not an object
    # from a python package that is in sys.path such as a sklearn object
    kit = "iris_old"
    ramp_kit_dir = os.path.join(PATH, "kits", kit)
    ramp_data_dir = os.path.join(PATH, "kits", kit)
    ramp_submission = os.path.join(PATH, "kits", kit, "submissions",
                                   "starting_kit")

    problem_module = import_module_from_source(
        os.path.join(ramp_kit_dir, 'problem.py'), 'problem')
    workflow = problem_module.workflow
    X_train, y_train = problem_module.get_train_data(path=ramp_data_dir)
    model = workflow.train_submission(ramp_submission, X_train, y_train)

    # test cloudpickle
    cloudpickle.dumps(model)
