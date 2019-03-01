import os
import pytest
from rampwf.hyperopt import Hyperparameter, run_hyperopt

PATH = os.path.dirname(__file__)


def test_hyperparameter():
    hp_1 = Hyperparameter(values=[1, 2, 3])
    assert hp_1.default == 1
    assert hp_1.n_values == 3
    assert hp_1.values == [1, 2, 3]
    assert hp_1.prior == [1. / 3, 1. / 3, 1. / 3]
    hp_2 = Hyperparameter(values=[1, 2, 3], prior=[0.1, 0.7, 0.2])
    assert hp_2.default == 1
    assert hp_2.n_values == 3
    assert hp_2.values == [1, 2, 3]
    assert hp_2.prior == [0.1, 0.7, 0.2]
    hp_3 = Hyperparameter(default=1)
    assert hp_3.default == 1
    assert hp_3.n_values == 1
    assert hp_3.values == [1]
    assert hp_3.prior == [1.0]
    with pytest.raises(ValueError) as e:
        Hyperparameter(values=[1, 2, 3], prior=[0.1, 0.7])
    assert str(e.value) == 'len(values) == 3 != 2 == len(prior)'
    with pytest.raises(ValueError) as e:
        Hyperparameter()
    assert str(e.value) == 'Either default or values must be defined.'
    with pytest.raises(ValueError) as e:
        Hyperparameter(values=[])
    assert str(e.value) == 'Values needs to contain at least one element.'
    with pytest.raises(ValueError) as e:
        Hyperparameter(default=2, values=[1])
    assert str(e.value) == 'Default must be among values.'


def test_hyperopt():
    ramp_kit_dir = os.path.join(
        PATH, 'interfaces', 'header_in_files', 'titanic')
    submission = 'starting_kit'
    run_hyperopt(ramp_kit_dir, submission, 'random', 10, is_cleanup=True)
