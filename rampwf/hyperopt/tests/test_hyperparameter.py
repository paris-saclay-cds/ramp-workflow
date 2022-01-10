import os
import pytest
from rampwf.hyperopt import Hyperparameter, run_hyperopt

PATH = os.path.dirname(__file__)


def test_hyperparameter():
    hp_1 = Hyperparameter(dtype='int', values=[1, 2, 3])
    assert hp_1.default == 1
    assert hp_1.n_values == 3
    assert list(hp_1.values) == [1, 2, 3]
    assert list(hp_1.prior) == [1. / 3, 1. / 3, 1. / 3]
    hp_2 = Hyperparameter(dtype='int', values=[1, 2, 3], prior=[0.1, 0.7, 0.2])
    assert hp_2.default == 1
    assert hp_2.n_values == 3
    assert list(hp_2.values) == [1, 2, 3]
    assert list(hp_2.prior) == [0.1, 0.7, 0.2]
    hp_3 = Hyperparameter(dtype='int', default=1)
    assert hp_3.default == 1
    assert hp_3.n_values == 1
    assert list(hp_3.values) == [1]
    assert list(hp_3.prior) == [1.0]
    with pytest.raises(ValueError) as e:
        Hyperparameter(dtype='int', values=[1, 2, 3], prior=[0.1, 0.7])
    assert str(e.value) == 'len(values) == 3 != 2 == len(prior)'
    with pytest.raises(ValueError) as e:
        Hyperparameter(dtype='int')
    assert str(e.value) == 'Either default or values must be defined.'
    with pytest.raises(ValueError) as e:
        Hyperparameter(dtype='int', values=[])
    assert str(e.value) == 'Values needs to contain at least one element.'
    with pytest.raises(ValueError) as e:
        Hyperparameter(dtype='int', default=2, values=[1])
    assert str(e.value) == 'Default must be among values.'


@pytest.mark.parametrize("submission", ['starting_kit', 'one_hyper_kit'])
def test_hyperopt_with_data_label(submission):
    # we can remove the creation of the folder
    # when only python >= 3.8 is supported by using
    # dirs_exist_ok
    ramp_kit_dir = os.path.join(
        PATH, 'interfaces', 'header_in_files', 'classifier_kit')
    destination_folder = \
        os.path.join(ramp_kit_dir, 'submissions',
                     'one_hyper_kit_titanic_hyperopt')
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    run_hyperopt(
        ramp_kit_dir, ramp_kit_dir,
        os.path.join(ramp_kit_dir, 'submissions'), 'titanic',
        submission, 'random', 64, True)


@pytest.mark.parametrize("submission", ['starting_kit', 'one_hyper_kit'])
def test_hyperopt_without_data_label(submission):
    ramp_kit_dir = os.path.join(
        PATH, 'interfaces', 'header_in_files', 'classifier_kit')
    run_hyperopt(
        ramp_kit_dir, ramp_kit_dir,
        os.path.join(ramp_kit_dir, 'submissions'), None,
        submission, 'random', 64, True)
