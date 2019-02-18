import pytest
from rampwf.hyperopt import Hyperparameter


def test_hyperparameter():
    hp_1 = Hyperparameter('hp_1', [1, 2, 3])
    assert hp_1.n_values == 3
    assert hp_1.values == [1, 2, 3]
    assert hp_1.prior == [1. / 3, 1. / 3, 1. / 3]
    hp_2 = Hyperparameter('hp_1', values=[1, 2, 3], prior=[0.1, 0.7, 0.2])
    assert hp_2.n_values == 3
    assert hp_2.values == [1, 2, 3]
    assert hp_2.prior == [0.1, 0.7, 0.2]
    with pytest.raises(ValueError) as e:
        Hyperparameter('hp_1', values=[1, 2, 3], prior=[0.1, 0.7])
    assert str(e.value) == 'len(values) == 3 != 2 == len(prior)'
