import numpy as np
import pandas as pd

from rampwf.workflows.ts_feature_extractor import extend_train_is


def test_extend_train_is_with_restart():
    X = np.arange(14)
    # the last episode is longer
    X_restart = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    X = np.concatenate(
        (X.reshape(-1, 1), X_restart.reshape(-1, 1)), axis=1)
    X_df = pd.DataFrame(data=X, columns=['col_1', 'restart'])
    n_burn_in = 1
    # check for first episode: train_is are the index of the associated y
    # with n_burn_in less timesteps than X for each episode
    # extended_train_is should be the indices of the first episode in X_df
    train_is = np.array([0, 1, 2])
    extended_train_is = extend_train_is(
        X_df, train_is, n_burn_in=n_burn_in, restart_name='restart')
    assert (extended_train_is == np.array([0, 1, 2, 3])).all()

    # check for second episode
    train_is = np.array([3, 4, 5])
    extended_train_is = extend_train_is(
        X_df, train_is, n_burn_in=n_burn_in, restart_name='restart')
    assert (extended_train_is == np.array([4, 5, 6, 7])).all()

    # check for first and last episodes
    train_is = np.array([0, 1, 2, 6, 7, 8, 9, 10])
    extended_train_is = extend_train_is(
        X_df, train_is, n_burn_in=n_burn_in, restart_name='restart')
    expected_output = np.array([0, 1, 2, 3, 8, 9, 10, 11, 12, 13])
    assert (extended_train_is == expected_output).all()

    # with n_burn_in = 2
    n_burn_in = 2
    train_is = np.array([0, 1, 4, 5, 6, 7])
    extended_train_is = extend_train_is(
        X_df, train_is, n_burn_in=n_burn_in, restart_name='restart')
    expected_output = np.array([0, 1, 2, 3, 8, 9, 10, 11, 12, 13])


def test_extend_train_is_no_burn_in_or_no_restart():

    train_is = np.arange(10)
    extended_train_is = extend_train_is(
        None, train_is, n_burn_in=0, restart_name='restart')
    assert (extended_train_is == train_is).all()

    extended_train_is = extend_train_is(
        None, train_is, n_burn_in=3, restart_name=None)
    expected_output = np.hstack((train_is, np.array([10, 11, 12])))
    assert (extended_train_is == expected_output).all()
